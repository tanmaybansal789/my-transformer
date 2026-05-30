import io
import os
import queue
import threading
import traceback
import contextlib
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from generate import Generator
from model import CharEncoder, Transformer, device
from train import Trainer


class QueueWriter(io.TextIOBase):
    def __init__(self, out_queue):
        self.out_queue = out_queue

    def write(self, s):
        if s:
            self.out_queue.put(s)
        return len(s)

    def flush(self):
        return


def _raise_keyboard_interrupt(thread):
    if thread is None or not thread.is_alive() or thread.ident is None:
        return False

    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident),
        ctypes.py_object(KeyboardInterrupt),
    )
    if result == 0:
        return False
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), None)
        raise RuntimeError("Failed to inject KeyboardInterrupt cleanly")
    return True


class GPTToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT Training + Generation Tool")
        self.root.geometry("1100x760")

        self.log_queue = queue.Queue()
        self.training_thread = None
        self.resume_checkpoint = None
        self.resume_checkpoint_path = None

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.train_frame = ttk.Frame(notebook, padding=12)
        self.generate_frame = ttk.Frame(notebook, padding=12)

        notebook.add(self.train_frame, text="Train")
        notebook.add(self.generate_frame, text="Generate")

        self._build_train_tab()
        self._build_generate_tab()

    def _build_train_tab(self):
        train_top = ttk.Frame(self.train_frame)
        train_top.pack(fill=tk.X, pady=(0, 10))

        self.input_path_var = tk.StringVar(value="input.txt")
        self.encoder_type_var = tk.StringVar(value="char")

        ttk.Label(train_top, text="Text").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_top, textvariable=self.input_path_var, width=70).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(train_top, text="Browse", command=self._pick_input_text).grid(row=1, column=1, sticky="ew")

        ttk.Label(train_top, text="Encoder").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(train_top, textvariable=self.encoder_type_var, 
                     values=["char", "r50k_base", "p50k_edit", "cl100k_base", "o200k_base"], 
                     state="readonly", 
                     width=18).grid(row=3, column=0, sticky="w")

        train_top.columnconfigure(0, weight=1)

        model_box = ttk.LabelFrame(self.train_frame, text="Model Config", padding=10)
        model_box.pack(fill=tk.X, pady=(0, 10))

        self.model_vars = {
            "max_len": tk.StringVar(value="256"),
            "d_embed": tk.StringVar(value="384"),
            "n_heads": tk.StringVar(value="6"),
            "d_hidden": tk.StringVar(value="1536"),
            "n_blocks": tk.StringVar(value="6"),
            "base": tk.StringVar(value="10000"),
        }
        self.model_entries = {}
        self._render_form_grid(model_box, self.model_vars, entry_store=self.model_entries)

        train_cfg_box = ttk.LabelFrame(self.train_frame, text="Training Config", padding=10)
        train_cfg_box.pack(fill=tk.X, pady=(0, 10))

        self.training_vars = {
            "batch_size": tk.StringVar(value="64"),
            "block_size": tk.StringVar(value="256"),
            "epochs": tk.StringVar(value="100"),
            "train_split": tk.StringVar(value="0.9"),
        }
        self._render_form_grid(train_cfg_box, self.training_vars)

        save_box = ttk.Frame(self.train_frame)
        save_box.pack(fill=tk.X, pady=(0, 10))
        self.custom_save_path_var = tk.StringVar(value="")
        ttk.Label(save_box, text="Optional checkpoint save path").grid(row=0, column=0, sticky="w")
        ttk.Entry(save_box, textvariable=self.custom_save_path_var, width=70).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(save_box, text="Browse", command=self._pick_save_path).grid(row=1, column=1, sticky="ew")
        save_box.columnconfigure(0, weight=1)

        resume_box = ttk.Frame(self.train_frame)
        resume_box.pack(fill=tk.X, pady=(0, 10))
        self.resume_path_var = tk.StringVar(value="")
        self.resume_status_var = tk.StringVar(value="No checkpoint loaded.")

        ttk.Label(resume_box, text="Resume from checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(resume_box, textvariable=self.resume_path_var, width=70).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(resume_box, text="Browse", command=self._pick_resume_checkpoint).grid(row=1, column=1, sticky="ew")
        ttk.Button(resume_box, text="Load", command=self._load_resume_checkpoint).grid(row=1, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(resume_box, text="Clear", command=self._clear_resume_checkpoint).grid(row=1, column=3, sticky="ew", padx=(6, 0))
        ttk.Label(resume_box, textvariable=self.resume_status_var).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        resume_box.columnconfigure(0, weight=1)

        controls = ttk.Frame(self.train_frame)
        controls.pack(fill=tk.X, pady=(0, 10))

        self.train_btn = ttk.Button(controls, text="Train!", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(controls, text="Stop training", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(controls, text="Clear logs", command=lambda: self.train_log.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        self.train_log = tk.Text(self.train_frame, height=18, wrap="word")
        self.train_log.pack(fill=tk.BOTH, expand=True)

    def _build_generate_tab(self):
        top = ttk.Frame(self.generate_frame)
        top.pack(fill=tk.X, pady=(0, 10))

        self.checkpoint_var = tk.StringVar(value="")
        ttk.Label(top, text="Checkpoint (.pt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.checkpoint_var, width=70).grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(top, text="Browse", command=self._pick_checkpoint).grid(row=1, column=1, sticky="ew")
        ttk.Button(top, text="Load", command=self._load_checkpoint).grid(row=1, column=2, sticky="ew", padx=(6, 0))
        top.columnconfigure(0, weight=1)

        prompt_box = ttk.LabelFrame(self.generate_frame, text="Prompt", padding=10)
        prompt_box.pack(fill=tk.BOTH, pady=(0, 10))

        self.prompt_text = tk.Text(prompt_box, height=6, wrap="word")
        self.prompt_text.pack(fill=tk.BOTH, expand=True)
        self.prompt_text.insert("1.0", "Hello, I am")

        params = ttk.Frame(self.generate_frame)
        params.pack(fill=tk.X, pady=(0, 10))

        self.max_tokens_var = tk.StringVar(value="300")
        self.temperature_var = tk.StringVar(value="1.0")

        ttk.Label(params, text="Max new tokens").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.max_tokens_var, width=12).grid(row=1, column=0, sticky="w", padx=(0, 20))

        ttk.Label(params, text="Temperature").grid(row=0, column=1, sticky="w")
        ttk.Entry(params, textvariable=self.temperature_var, width=12).grid(row=1, column=1, sticky="w")

        controls = ttk.Frame(self.generate_frame)
        controls.pack(fill=tk.X, pady=(0, 10))
        self.generate_btn = ttk.Button(controls, text="Generate text", command=self._generate_text)
        self.generate_btn.pack(side=tk.LEFT)

        ttk.Button(controls, text="Clear output", command=lambda: self.output_text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=(8, 0))

        self.output_text = tk.Text(self.generate_frame, height=18, wrap="word")
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.loaded_model = None
        self.loaded_encoder = None

    def _render_form_grid(self, parent, fields, entry_store=None):
        for idx, (name, var) in enumerate(fields.items()):
            row = idx // 3
            col = idx % 3
            group = ttk.Frame(parent)
            group.grid(row=row, column=col, sticky="ew", padx=6, pady=4)
            ttk.Label(group, text=name).pack(anchor="w")
            entry = ttk.Entry(group, textvariable=var, width=16)
            entry.pack(anchor="w")
            if entry_store is not None:
                entry_store[name] = entry
            parent.columnconfigure(col, weight=1)

    def _pick_input_text(self):
        path = filedialog.askopenfilename(title="Select Training Text")
        if path:
            self.input_path_var.set(path)

    def _pick_save_path(self):
        path = filedialog.asksaveasfilename(
            title="Save Checkpoint As",
            defaultextension=".pt",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
        )
        if path:
            self.custom_save_path_var.set(path)

    def _pick_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
        )
        if path:
            self.checkpoint_var.set(path)

    def _pick_resume_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select Resume Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
        )
        if path:
            self.resume_path_var.set(path)

    def _set_model_config_state(self, state):
        for entry in self.model_entries.values():
            entry.config(state=state)

    def _load_resume_checkpoint(self):
        path = self.resume_path_var.get().strip()
        if not path:
            messagebox.showerror("Missing Checkpoint", "Choose a resume checkpoint first.")
            return

        if not os.path.exists(path):
            messagebox.showerror("Missing File", f"Resume checkpoint not found:\n{path}")
            return

        try:
            checkpoint = torch.load(path, map_location="cpu")
            required = ["model_state_dict", "model_config", "encoder_type", "text"]
            for key in required:
                if key not in checkpoint:
                    raise ValueError(f"Invalid checkpoint: missing '{key}'")

            loaded_model_config = checkpoint["model_config"]
            for key in self.model_vars:
                if key in loaded_model_config:
                    self.model_vars[key].set(str(loaded_model_config[key]))

            self.encoder_type_var.set(checkpoint["encoder_type"])
            self._set_model_config_state("disabled")

            self.resume_checkpoint = checkpoint
            self.resume_checkpoint_path = path
            self.resume_status_var.set(f"Loaded: {os.path.basename(path)}")
            self._log(f"Loaded resume checkpoint: {path}\n")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _clear_resume_checkpoint(self):
        self.resume_checkpoint = None
        self.resume_checkpoint_path = None
        self.resume_path_var.set("")
        self.resume_status_var.set("No checkpoint loaded.")
        self._set_model_config_state("normal")
        self._log("Cleared resume checkpoint. Model settings are editable again.\n")

    def _log(self, text):
        self.log_queue.put(text)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            text = self.log_queue.get()
            self.train_log.insert(tk.END, text)
            self.train_log.see(tk.END)
        self.root.after(100, self._poll_log_queue)

    def _collect_training_config(self):
        model_config = {
            "n_vocab": None,
            "max_len": int(self.model_vars["max_len"].get()),
            "d_embed": int(self.model_vars["d_embed"].get()),
            "n_heads": int(self.model_vars["n_heads"].get()),
            "d_hidden": int(self.model_vars["d_hidden"].get()),
            "n_blocks": int(self.model_vars["n_blocks"].get()),
            "base": int(self.model_vars["base"].get()),
        }

        training_config = {
            "batch_size": int(self.training_vars["batch_size"].get()),
            "block_size": int(self.training_vars["block_size"].get()),
            "epochs": int(self.training_vars["epochs"].get()),
            "train_split": float(self.training_vars["train_split"].get()),
        }

        if not (0.0 < training_config["train_split"] < 1.0):
            raise ValueError("train_split must be between 0 and 1")

        return model_config, training_config

    def _start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training Running", "A training job is already running.")
            return

        input_path = self.input_path_var.get().strip()
        if not input_path and self.resume_checkpoint is None:
            messagebox.showerror("Missing Input", "Please choose a training text file or load a resume checkpoint.")
            return

        if input_path and not os.path.exists(input_path):
            messagebox.showerror("Missing File", f"Training file not found:\n{input_path}")
            return

        try:
            model_config, training_config = self._collect_training_config()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        encoder_type = self.encoder_type_var.get().strip() or "char"
        custom_save_path = self.custom_save_path_var.get().strip()
        resume_checkpoint = self.resume_checkpoint

        text = None
        if input_path:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()

        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log("Training started \n")
        self._log(f"Using device: {device}\n")

        def train_worker():
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    if resume_checkpoint is not None:
                        resume_model_config = resume_checkpoint["model_config"]
                        resume_model = Transformer(**resume_model_config)
                        resume_model.load_state_dict(resume_checkpoint["model_state_dict"])

                        training_text = text if text is not None else resume_checkpoint["text"]

                        trainer = Trainer(
                            encoder_type=resume_checkpoint["encoder_type"],
                            text=training_text,
                            model_config=resume_model_config,
                            training_config=training_config,
                            model=resume_model,
                        )

                        if "optimizer_state_dict" in resume_checkpoint:
                            trainer.optimiser.load_state_dict(resume_checkpoint["optimizer_state_dict"])

                        self._log(f"Resuming training from: {self.resume_checkpoint_path}\n")
                    else:
                        trainer = Trainer(
                            encoder_type=encoder_type,
                            text=text,
                            model_config=model_config,
                            training_config=training_config,
                        )

                    if custom_save_path:
                        checkpoint = trainer.train(save_checkpoint=False)
                        torch.save(checkpoint, custom_save_path)
                        self._log(f"Saved checkpoint to: {custom_save_path}\n")
                    else:
                        trainer.train(save_checkpoint=True)
                        self._log("Saved checkpoint with default filename.\n")

                self._log("Training completed\n")
            except Exception:
                self._log("Training failed\n")
                self._log(traceback.format_exc() + "\n")
            finally:
                self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

        self.training_thread = threading.Thread(target=train_worker, daemon=True)
        self.training_thread.start()

    def _stop_training(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            messagebox.showinfo("No Active Training", "There is no active training job to stop.")
            self.stop_btn.config(state=tk.DISABLED)
            return

        try:
            did_raise = _raise_keyboard_interrupt(self.training_thread)
            if did_raise:
                self._log("Stop requested.\n")
                self.stop_btn.config(state=tk.DISABLED)
            else:
                self._log("Stop request failed: training thread is no longer active.\n")
                self.stop_btn.config(state=tk.DISABLED)
        except Exception as exc:
            messagebox.showerror("Stop Failed", str(exc))

    def _load_checkpoint(self):
        path = self.checkpoint_var.get().strip()
        if not path:
            messagebox.showerror("Missing Checkpoint", "Choose a checkpoint file first.")
            return

        if not os.path.exists(path):
            messagebox.showerror("Missing File", f"Checkpoint not found:\n{path}")
            return

        try:
            checkpoint = torch.load(path, map_location=device)
            model = Transformer(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            if checkpoint["encoder_type"] == "char":
                encoder = CharEncoder(checkpoint["text"])
            else:
                from model import tiktoken

                encoder = tiktoken.get_encoding(checkpoint["encoder_type"])

            self.loaded_model = model
            self.loaded_encoder = encoder

            self.output_text.insert(tk.END, f"Loaded checkpoint: {path}\n")
            self.output_text.see(tk.END)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _generate_text(self):
        if self.loaded_model is None or self.loaded_encoder is None:
            messagebox.showerror("No Model", "Load a checkpoint before generating text.")
            return

        prompt = self.prompt_text.get("1.0", tk.END).rstrip("\n")
        if not prompt:
            messagebox.showerror("Missing Prompt", "Please enter a prompt.")
            return

        try:
            max_new_tokens = int(self.max_tokens_var.get())
            temperature = float(self.temperature_var.get())
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be > 0")
            if temperature <= 0:
                raise ValueError("temperature must be > 0")
        except Exception as exc:
            messagebox.showerror("Invalid Generate Params", str(exc))
            return

        self.generate_btn.config(state=tk.DISABLED)
        self.output_text.insert(tk.END, "\nGenerating...\n")
        self.output_text.see(tk.END)

        def generate_worker():
            try:
                generator = Generator(self.loaded_model, self.loaded_encoder)

                x = torch.tensor(
                    generator.encoder.encode(prompt),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                for _ in range(max_new_tokens):
                    next_token_id = generator.generate_next(
                        x=x,
                        temperature=temperature,
                    )

                    x = torch.cat([x, next_token_id], dim=1)
                    next_token = generator.encoder.decode([next_token_id.item()])
                    self.root.after(0, lambda token=next_token: self._append_generation(token))

            except Exception as exc:
                err_msg = str(exc)
                self.root.after(0, lambda msg=err_msg: messagebox.showerror("Generation Error", msg))
            finally:
                self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))

        threading.Thread(target=generate_worker, daemon=True).start()

    def _append_generation(self, generated):
        self.output_text.insert(tk.END, generated)
        self.output_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = GPTToolApp(root)
    root.mainloop()