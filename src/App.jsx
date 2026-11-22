import React, { useState, useEffect, useMemo, useRef } from "react";

const PIPELINE_STAGES = [
  {
    "id": 1,
    "name": "1. Tokenizer",
    "sub": "Text -> Token IDs",
    "phaseId": "phase-3",
    "badge": "Phase 3 • Tokenization",
    "desc": "Raw text is split into subword integer IDs using Byte-Pair Encoding: \"The capital of France is\" -> [464, 3139, 286, 4881, 318]."
  },
  {
    "id": 2,
    "name": "2. Embeddings",
    "sub": "IDs -> Dense Vectors",
    "phaseId": "phase-3",
    "badge": "Phase 3 • Vector Space",
    "desc": "Each token ID looks up a learned vector (e.g., 768 floats) plus positional info (RoPE/Positional Embedding) so order matters."
  },
  {
    "id": 3,
    "name": "3. Self-Attention",
    "sub": "Context Mixing (Q·Kᵀ)",
    "phaseId": "phase-4",
    "badge": "Phase 4 • Heart of GPT",
    "desc": "Tokens communicate via Queries, Keys, and Values with a causal mask so 'France' attends back to 'capital' without peeking at future words."
  },
  {
    "id": 4,
    "name": "4. Feed-Forward",
    "sub": "MLP Knowledge Store",
    "phaseId": "phase-4",
    "badge": "Phase 4 • Transformer Block",
    "desc": "Each token passes through LayerNorm/RMSNorm, a 2-layer MLP with GELU/SwiGLU activation, and residual skip connections."
  },
  {
    "id": 5,
    "name": "5. LM Head + Softmax",
    "sub": "Logits -> Probabilities",
    "phaseId": "phase-5",
    "badge": "Phase 5 • Next-Token Prediction",
    "desc": "Final hidden vectors project back to vocabulary size (e.g., 50,257 logits). Softmax turns them into probabilities—'Paris' scores highest!"
  },
  {
    "id": 6,
    "name": "6. SFT & Local Run",
    "sub": "Assistant Alignment",
    "phaseId": "phase-6",
    "badge": "Phase 6 • Post-Training",
    "desc": "Fine-tune the base predictor on Q&A instructions (SFT/LoRA), quantize weights, and serve locally via Ollama or FastAPI."
  }
];

const SETUP_SOFTWARES = [
  {
    "name": "Python 3.11 (64-bit)",
    "url": "https://www.python.org/downloads/",
    "role": "Core language runtime for writing models from scratch, training loops, and data pipelines.",
    "whereUsed": "Used across every AI lab (OpenAI, Meta, Hugging Face) as the control plane orchestrating C++/CUDA kernels.",
    "whatYouBuild": "Your isolated .venv environment, BPE tokenizer, GPT model classes, training loop, and API server.",
    "winget": "winget install -e --id Python.Python.3.11"
  },
  {
    "name": "Visual Studio Code",
    "url": "https://code.visualstudio.com/",
    "role": "Code editor, integrated terminal, Jupyter notebook runner, and tensor shape debugger.",
    "whereUsed": "Daily IDE for ML engineers to inspect tensor shapes, debug training runs, and browse repositories.",
    "whatYouBuild": "Modular Python scripts (tokenizer.py, model.py, train.py) and interactive notebooks.",
    "winget": "winget install -e --id Microsoft.VisualStudioCode"
  },
  {
    "name": "Git for Windows",
    "url": "https://git-scm.com/downloads/win",
    "role": "Version control for tracking model experiments, checkpoint scripts, and pushing to GitHub.",
    "whereUsed": "Every collaborative AI codebase and open-source model repository on GitHub/Hugging Face.",
    "whatYouBuild": "Clean commit history tracking every phase from micrograd to your pretrained mini-GPT.",
    "winget": "winget install -e --id Git.Git"
  },
  {
    "name": "Ollama",
    "url": "https://ollama.com/download",
    "role": "Runs quantized open-source LLMs (Llama 3, Qwen, Gemma) locally on your CPU or GPU.",
    "whereUsed": "Local developer workstations and private on-premise enterprise AI deployments.",
    "whatYouBuild": "A local reference baseline to compare latency and answer quality against your custom GPT.",
    "winget": "winget install -e --id Ollama.Ollama"
  }
];

const SETUP_PACKAGES = [
  {
    "name": "torch",
    "desc": "PyTorch autograd & GPU tensor engine",
    "whereUsed": "Meta, OpenAI & Tesla AI training clusters",
    "whatYouBuild": "Self-attention, Transformer blocks & GPTModel"
  },
  {
    "name": "numpy",
    "desc": "Fast C-backed matrix & vector math",
    "whereUsed": "Every scientific & ML preprocessing pipeline",
    "whatYouBuild": "Matrix multiplication & Softmax visualizers"
  },
  {
    "name": "transformers",
    "desc": "Reference architectures & weight loaders",
    "whereUsed": "Hugging Face ecosystem & fine-tuning pipelines",
    "whatYouBuild": "Weight loader mapping GPT-2 into your code"
  },
  {
    "name": "tokenizers",
    "desc": "Rust-accelerated subword BPE tokenizers",
    "whereUsed": "Production LLM inference engines (vLLM, TGI)",
    "whatYouBuild": "Benchmark comparison for your pure-Python BPE"
  },
  {
    "name": "datasets",
    "desc": "Streaming text corpora & dataset loaders",
    "whereUsed": "Pretraining & SFT data curation pipelines",
    "whatYouBuild": "TinyStories & instruction Q&A data loaders"
  },
  {
    "name": "jupyterlab",
    "desc": "Interactive notebooks for step-by-step math",
    "whereUsed": "Research prototyping & activation diagnostics",
    "whatYouBuild": "Interactive attention heatmap & loss notebooks"
  },
  {
    "name": "pandas",
    "desc": "Tabular dataset cleaning & inspection",
    "whereUsed": "Filtering & deduplicating SFT instruction data",
    "whatYouBuild": "Clean Q&A instruction formatting pipelines"
  },
  {
    "name": "matplotlib",
    "desc": "Plotting loss curves & attention heatmaps",
    "whereUsed": "Training run diagnostics & paper figures",
    "whatYouBuild": "Train vs. validation loss & cosine LR plots"
  },
  {
    "name": "pytest",
    "desc": "Automated shape & gradient unit testing",
    "whereUsed": "CI/CD checks in production ML repositories",
    "whatYouBuild": "Automated tensor-shape & prompt quality tests"
  },
  {
    "name": "mlflow",
    "desc": "Experiment tracking for hyperparams & loss",
    "whereUsed": "Enterprise ML teams comparing training runs",
    "whatYouBuild": "Local experiment dashboard for LR & loss runs"
  },
  {
    "name": "fastapi",
    "desc": "Typed Python REST API framework",
    "whereUsed": "Production model-serving microservices",
    "whatYouBuild": "/v1/generate HTTP endpoint for your GPT"
  },
  {
    "name": "uvicorn",
    "desc": "High-speed ASGI web server for FastAPI",
    "whereUsed": "Hosting Python AI APIs in containers",
    "whatYouBuild": "Localhost server running your custom LLM API"
  },
  {
    "name": "requests",
    "desc": "HTTP client for downloading corpora & testing APIs",
    "whereUsed": "Fetching datasets & querying inference endpoints",
    "whatYouBuild": "Client script testing your FastAPI & Ollama APIs"
  },
  {
    "name": "lxml",
    "desc": "Fast XML/HTML text extraction & parsing",
    "whereUsed": "Scraping & cleaning raw web text for pretraining",
    "whatYouBuild": "Clean corpus extractor stripping HTML tags"
  }
];

const SETUP_COMMANDS_TEXT = "mkdir VIT-Automation\ncd VIT-Automation\npy -3.11 -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\npython -m pip install numpy pandas matplotlib jupyterlab transformers datasets tokenizers pytest lxml requests mlflow fastapi uvicorn\npython -m pip install torch\npython -c \"import torch, transformers, datasets, mlflow, fastapi; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\"";

const ROADMAP_PHASES = [
  {
    "id": "phase-1",
    "number": "01",
    "category": "foundations",
    "accent": "teal",
    "duration": "Week 1–2",
    "title": "Python, NumPy & PyTorch Foundations for LLMs",
    "subtitle": "Master clean Python OOP, NumPy broadcasting, the (B, T, C) tensor shape mental model, and probability/loss math before writing neural nets.",
    "plainBigPicture": "An LLM is really just a giant calculator that reads numbers representing words and predicts the probability of the next word. Before building the brain, you need Python (the glue language), NumPy (a fast math library for grids of numbers), and PyTorch (an AI framework that runs those grids on your GPU).",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 1 Skills",
      "summary": "Every production LLM API (OpenAI, Anthropic, Ollama) relies directly on these exact primitives on every single request:",
      "examples": [
        "OpenAI's 'logprobs' API parameter exposes the exact raw Logits -> Softmax probability percentages you build in Module 1.5.",
        "Production inference servers batch multiple users' prompts together into a single (B, T, C) 3D tensor (Module 1.3) to maximize GPU throughput."
      ]
    },
    "jargonDecoder": [
      {
        "term": "LLM",
        "def": "Next-word probability predictor"
      },
      {
        "term": "NumPy",
        "def": "Fast grid-of-numbers math library"
      },
      {
        "term": "PyTorch",
        "def": "AI framework with GPU acceleration"
      },
      {
        "term": "Tensor",
        "def": "Multi-dimensional spreadsheet of numbers"
      },
      {
        "term": "Logits",
        "def": "Raw unscaled word scores"
      },
      {
        "term": "Softmax",
        "def": "Converts raw scores into % probabilities"
      }
    ],
    "achievements": [
      "Confidently write clean, modular Python classes and @dataclass configs inside an isolated .venv environment.",
      "Trace (Batch, Sequence, Embedding) 3D tensor shapes across matrix multiplications without shape-mismatch errors.",
      "Convert raw model logits into calibrated token probabilities using Softmax and compute Cross-Entropy loss."
    ],
    "miniProject": "Build a Matrix Multiplication & Softmax Temperature Visualizer in NumPy/PyTorch",
    "topics": [
      {
        "id": "p1-t1",
        "codeNum": "1.1",
        "animType": "python_oop",
        "title": "Python Data Structures, OOP & Clean ML Code",
        "duration": "2 Days",
        "plainEnglish": "Python is the programming language we use to orchestrate everything. Because an AI model has hundreds of configurable knobs (layers, heads, vocabulary size), we bundle them cleanly inside Python 'Classes' (reusable blueprints) and isolate our tools in a '.venv' folder using 'pip' so nothing breaks your computer.",
        "liveExample": "Hugging Face's Transformers library defines every model (LlamaConfig, GPT2Config) using this exact Python @dataclass + OOP blueprint pattern.",
        "jargonTerms": [
          {
            "term": "Python 3.11",
            "def": "The human-readable programming language used to write AI scripts."
          },
          {
            "term": ".venv (Virtual Env)",
            "def": "An isolated sandbox folder holding project packages so different projects never clash."
          },
          {
            "term": "pip",
            "def": "Python's built-in package installer used to download libraries like PyTorch."
          },
          {
            "term": "OOP (Classes)",
            "def": "Grouping data and functions together into reusable blueprints."
          },
          {
            "term": "@dataclass",
            "def": "A clean Python shortcut for creating settings/config containers without boilerplate."
          }
        ],
        "why": "Every PyTorch model is built by subclassing classes, overriding forward(), and managing token dictionaries and configuration dataclasses cleanly.",
        "lessonSections": [
          {
            "heading": "1. Why Virtual Environments (.venv) Come First",
            "body": "AI libraries evolve rapidly—one project may need PyTorch 2.2 while another needs 2.5. Creating a .venv folder gives your project its own private Python interpreter and site-packages folder.",
            "codeExample": "py -3.11 -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install torch numpy"
          },
          {
            "heading": "2. Configuration Blueprints with @dataclass",
            "body": "Instead of passing 12 separate arguments into every layer constructor, LLM codebases store architecture hyperparameters in a single @dataclass object.",
            "codeExample": "from dataclasses import dataclass\n\n@dataclass\nclass GPTConfig:\n    vocab_size: int = 50257\n    context_length: int = 256\n    emb_dim: int = 384\n    n_heads: int = 6\n    n_layers: int = 6"
          }
        ],
        "concepts": [
          "Isolated Python 3.11 virtual environments (.venv), pip dependency locking, and JupyterLab cell debugging.",
          "Python classes, inheritance (super().__init__()), and dunder methods (__call__, __len__, __getitem__, __repr__).",
          "Clean configuration management with @dataclass, type hints, and generator expressions."
        ],
        "code": "from dataclasses import dataclass\n\n@dataclass\nclass GPTConfig:\n    vocab_size: int = 50257\n    context_length: int = 256\n    emb_dim: int = 384\n    n_heads: int = 6\n    n_layers: int = 6\n\ncfg = GPTConfig()\nprint(f\"Model width: {cfg.emb_dim}, Head dim: {cfg.emb_dim // cfg.n_heads}\")",
        "quiz": {
          "question": "Why do LLM codebases use @dataclass config objects instead of passing individual integers to every layer?",
          "options": [
            "It makes PyTorch run 10x faster on CPUs",
            "It centralizes hyperparameters (emb_dim, n_heads, n_layers) cleanly so every module stays synchronized",
            "It automatically downloads pretrained weights from Hugging Face"
          ],
          "answerIndex": 1,
          "explanation": "Centralizing hyperparameters in a @dataclass guarantees that embedding layers, attention heads, and MLP blocks all share consistent dimensions."
        },
        "pitfall": "Never install ML packages globally into your OS Python—always verify 'Get-Command python' points inside your project's .venv folder.",
        "project": "Write a clean TokenizerBase Python class with __len__, encode(), and decode() methods plus a @dataclass config object.",
        "resourceTitle": "Sebastian Raschka — Build an LLM From Scratch (Environment & Python Setup)",
        "resourceNote": "Start here to structure clean Python classes and virtual environments before writing PyTorch models.",
        "links": [
          {
            "type": "YouTube",
            "label": "Sebastian Raschka — Build an LLM from Scratch 1: Set Up Your Code Environment",
            "url": "https://www.youtube.com/watch?v=yAcWnfsZhzo"
          },
          {
            "type": "Community Guide",
            "label": "Real Python — Object-Oriented Programming & Data Classes Guide",
            "url": "https://realpython.com/python-data-classes/"
          }
        ]
      },
      {
        "id": "p1-t2",
        "codeNum": "1.2",
        "animType": "broadcasting",
        "title": "Vectors, Matrices & Broadcasting in NumPy",
        "duration": "3 Days",
        "plainEnglish": "Standard Python lists are far too slow for AI because Python checks data types one item at a time. NumPy stores numbers in contiguous C-memory blocks so your CPU can multiply millions of numbers simultaneously.",
        "liveExample": "Vector search engines (Pinecone, Chroma, pgvector) powering RAG chatbots find relevant documents using the exact dot-product cosine similarity math here.",
        "jargonTerms": [
          {
            "term": "NumPy",
            "def": "Numerical Python—a C-powered library for ultra-fast array math."
          },
          {
            "term": "Vector",
            "def": "A 1D list of numbers representing a word's meaning."
          },
          {
            "term": "Matrix",
            "def": "A 2D grid (rows x columns) of numbers."
          },
          {
            "term": "Broadcasting",
            "def": "Automatically stretching a smaller row/column to add onto a bigger table."
          }
        ],
        "why": "Token embeddings are vectors, weight layers are matrices, and NumPy broadcasting rules govern how biases and attention masks combine automatically.",
        "lessonSections": [
          {
            "heading": "1. Dot Products Measure Semantic Similarity",
            "body": "When two word vectors point in a similar direction, their dot product (A @ B) is large and positive. Self-attention uses this exact operation to score how strongly two words relate.",
            "codeExample": "import numpy as np\nking = np.array([0.9, 0.8, 0.1])\nqueen = np.array([0.88, 0.85, 0.12])\nprint('Similarity score:', np.dot(king, queen))"
          }
        ],
        "concepts": [
          "Scalars (0D), Vectors (1D), Matrices (2D), and Tensors (3D/4D) as nested numeric arrays.",
          "Element-wise operations vs. dot products (np.dot / @) as geometric cosine similarity.",
          "Broadcasting rules: aligning trailing dimensions (e.g., adding a (1, C) bias vector to a (T, C) token matrix)."
        ],
        "code": "import numpy as np\n\nX = np.array([[1.0, 0.2, -0.5],\n              [0.9, 0.1, -0.4],\n              [-0.8, 0.7, 0.3],\n              [0.0, -0.5, 0.9]])\n\nbias = np.array([0.1, -0.1, 0.05])\nprint(\"Broadcasted sum shape:\", (X + bias).shape)\nprint(\"Pairwise similarity (4x4):\\n\", np.round(X @ X.T, 2))",
        "quiz": {
          "question": "If X has shape (4, 3) and bias has shape (3,), what happens when you compute X + bias?",
          "options": [
            "NumPy raises a ValueError because shapes don't match",
            "NumPy broadcasts bias across all 4 rows without copying memory, producing shape (4, 3)",
            "NumPy truncates X to shape (3,)"
          ],
          "answerIndex": 1,
          "explanation": "Trailing dimensions match (3 == 3), so NumPy virtually stretches the 1D bias vector across all 4 rows."
        },
        "pitfall": "Confusing element-wise multiplication (*) with matrix multiplication (@). In Python, A * B multiplies matching cells; A @ B performs row-by-column dot products.",
        "project": "Implement vector cosine similarity in NumPy and test how similar word vectors have dot products near +1.",
        "resourceTitle": "3Blue1Brown — Essence of Linear Algebra (Chapters 1 & 4)",
        "resourceNote": "Visualizes vectors, linear transformations, matrix multiplication, and dot products geometrically.",
        "links": [
          {
            "type": "YouTube",
            "label": "3Blue1Brown — Vectors | Essence of Linear Algebra Ch. 1",
            "url": "https://www.youtube.com/watch?v=fNk_zzaMoSs"
          },
          {
            "type": "Community Guide",
            "label": "Jay Alammar — A Visual Intro to NumPy and Data Representation",
            "url": "https://jalammar.github.io/visual-numpy/"
          }
        ]
      },
      {
        "id": "p1-t3",
        "codeNum": "1.3",
        "animType": "tensor_btc",
        "title": "Matrix Multiplication & Tensor Shapes (B, T, C) Mental Model",
        "duration": "3 Days",
        "plainEnglish": "Think of a 'Tensor' as a stack of spreadsheets. Throughout an entire LLM, data travels in a 3D box shaped (B, T, C): B is Batch size, T is Tokens in each sentence, and C is Channels (numbers describing each token).",
        "liveExample": "In Llama-3 8B, every prompt flows through 32 Transformer layers with shape (B, T, 4096) before projecting to 128,256 vocabulary logits.",
        "jargonTerms": [
          {
            "term": "Tensor",
            "def": "A generalized container of numbers (1D list, 2D table, or 3D+ cube)."
          },
          {
            "term": "B (Batch Size)",
            "def": "Number of separate sentences processed in parallel for speed."
          },
          {
            "term": "T (Sequence Length)",
            "def": "Number of tokens (word-pieces) inside a single sentence."
          },
          {
            "term": "C (Embedding Dim)",
            "def": "Size of the number list describing each individual token."
          }
        ],
        "why": "Every single Transformer layer passes around a 3D tensor of shape (Batch, Time/Sequence, Channels/Embedding). Knowing this mental model prevents 90% of bugs.",
        "lessonSections": [
          {
            "heading": "1. Tracking (B, T, C) Through Linear Layers",
            "body": "PyTorch's nn.Linear(C, D) always operates on the LAST axis (-1), leaving B and T completely untouched.",
            "codeExample": "import torch, torch.nn as nn\nx = torch.randn(2, 5, 64)      # (B=2, T=5, C=64)\nproj = nn.Linear(64, 128)\nprint(proj(x).shape)           # -> torch.Size([2, 5, 128])"
          }
        ],
        "concepts": [
          "What B (batch size), T (sequence length / context window), and C (embedding dimension) represent physically.",
          "Batched matrix multiplication: (B, T, C) @ (C, D) -> (B, T, D) transforms every token independently in parallel.",
          "Reshaping vs. transposing: why .view()/.reshape() and .transpose(-2, -1) behave differently in memory."
        ],
        "code": "import torch\n\nB, T, C = 2, 5, 64\nW_out = 128\nx = torch.randn(B, T, C)\nW = torch.randn(C, W_out)\ny = x @ W\nprint(\"Transformed tensor shape:\", y.shape)",
        "quiz": {
          "question": "If tensor x has shape (4, 16, 128) and weight matrix W has shape (128, 256), what is the shape of x @ W?",
          "options": [
            "(4, 16, 256)",
            "(4, 256, 16)",
            "(16, 128, 256)"
          ],
          "answerIndex": 0,
          "explanation": "Batched matrix multiplication contracts the inner dimension 128 and preserves the batch (4) and sequence (16) dimensions."
        },
        "pitfall": "Never use .view() to swap axes like (B, T, C) to (B, C, T)—that scrambles data! Always use .transpose(1, 2) when swapping dimensions.",
        "project": "Write a shape-tracing script that passes a dummy (4, 16, 128) tensor through projections, splits it into 4 heads (4, 4, 16, 32), and merges it back.",
        "resourceTitle": "Jay Alammar — The Illustrated Transformer (Tensor Dimension Diagrams)",
        "resourceNote": "Study how input matrices keep their sequence length T constant from bottom to top of the stack.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Jay Alammar — The Illustrated Transformer (Must-Read Visual Blog)",
            "url": "https://jalammar.github.io/illustrated-transformer/"
          },
          {
            "type": "YouTube",
            "label": "3Blue1Brown — Transformers, the tech behind LLMs (Ch. 5)",
            "url": "https://www.youtube.com/watch?v=wjZofJX0v4M"
          }
        ]
      },
      {
        "id": "p1-t4",
        "codeNum": "1.4",
        "animType": "tensor_btc",
        "title": "PyTorch Tensors, Device Management & Vectorization",
        "duration": "2 Days",
        "plainEnglish": "PyTorch is like NumPy with two superpowers: it runs calculations on a GPU 50x faster than a CPU, and it automatically tracks gradients for learning.",
        "liveExample": "Modern LLM training clusters run PyTorch in bfloat16 precision across thousands of NVIDIA H100 GPUs using identical .to(device) tensor APIs.",
        "jargonTerms": [
          {
            "term": "PyTorch",
            "def": "Meta's open-source Python library for building and training neural networks."
          },
          {
            "term": "CUDA / GPU",
            "def": "Graphics card hardware designed to do thousands of matrix multiplications at once."
          },
          {
            "term": "Vectorization",
            "def": "Replacing slow one-by-one Python loops with single-step array math."
          }
        ],
        "why": "PyTorch replaces slow Python for-loops with vectorized C++/CUDA tensor kernels and lets the exact same code run on CPU, NVIDIA CUDA, or Apple MPS.",
        "lessonSections": [
          {
            "heading": "1. Device-Agnostic Tensor Execution",
            "body": "Always move both your model and your input batches to the same device variable so your code runs seamlessly on laptops and cloud GPUs.",
            "codeExample": "device = 'cuda' if torch.cuda.is_available() else 'cpu'\nx = torch.randn(32, 128, device=device)"
          }
        ],
        "concepts": [
          "Creating tensors, inspecting .shape, .dtype (float32, bfloat16, int64), and .device.",
          "Device-agnostic code pattern: device = 'cuda' if torch.cuda.is_available() else 'cpu'.",
          "Vectorization: replacing nested Python loops with fast tensor reductions (.sum(dim=-1), .mean(dim=-1, keepdim=True))."
        ],
        "code": "import torch\n\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nx = torch.randn(32, 128, device=device, dtype=torch.float32)\nmean = x.mean(dim=-1, keepdim=True)\nvar = x.var(dim=-1, keepdim=True, unbiased=False)\nprint(f\"Running on {device} | Mean shape: {mean.shape}\")",
        "quiz": {
          "question": "Why do we pass keepdim=True when computing x.mean(dim=-1, keepdim=True) on a (32, 128) tensor?",
          "options": [
            "To keep the output shape as (32, 1) so it broadcasts cleanly back against (32, 128)",
            "To prevent PyTorch from deleting the input tensor",
            "To force CPU execution"
          ],
          "answerIndex": 0,
          "explanation": "Without keepdim=True, (32, 128).mean(dim=-1) collapses to (32,), which broadcasts along the wrong dimension if subtracted from (32, 128)."
        },
        "pitfall": "Forgetting keepdim=True during reductions! Without keepdim=True, (32, 128).mean(dim=-1) becomes (32,) instead of (32, 1), causing silent wrong broadcasting.",
        "project": "Benchmark a pure-Python nested loop dot-product against vectorized PyTorch tensor operations on 1,000x1,000 matrices.",
        "resourceTitle": "PyTorch Official Tutorials — Introduction to Tensors",
        "resourceNote": "Covers tensor creation, data types, GPU device transfers, and essential tensor ops.",
        "links": [
          {
            "type": "YouTube",
            "label": "PyTorch Official — Introduction to PyTorch Tensors",
            "url": "https://www.youtube.com/watch?v=r7QDUPb2dCM"
          },
          {
            "type": "Blog / Docs",
            "label": "PyTorch Official Tutorial — Introduction to Tensors",
            "url": "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html"
          }
        ]
      },
      {
        "id": "p1-t5",
        "codeNum": "1.5",
        "animType": "softmax_logits",
        "title": "Probability Distributions, Logits, Softmax & Cross-Entropy Loss",
        "duration": "3 Days",
        "plainEnglish": "When an LLM guesses the next word, it outputs a raw score ('Logit') for every word in its dictionary. 'Softmax' turns those scores into percentages adding up to 100%, and 'Cross-Entropy Loss' grades how wrong the guess was.",
        "liveExample": "Every token generated by ChatGPT or Claude comes from sampling this Softmax probability distribution over 100,000+ vocabulary logits.",
        "jargonTerms": [
          {
            "term": "Logits",
            "def": "Raw positive or negative scores before converting to percentages."
          },
          {
            "term": "Softmax",
            "def": "Math formula turning any list of numbers into probabilities between 0% and 100%."
          },
          {
            "term": "Cross-Entropy Loss",
            "def": "Penalty score: 0 means 100% confident & right; high means confident & wrong."
          }
        ],
        "why": "An LLM outputs unnormalized raw scores (logits) across 50,000+ vocabulary words. Softmax turns them into probabilities, and Cross-Entropy grades how confident the model was in the true next token.",
        "lessonSections": [
          {
            "heading": "1. Initial Loss Sanity Check: ln(vocab_size)",
            "body": "Before training begins, a well-initialized model assigns roughly equal probability 1/V to every word. Thus initial Cross-Entropy loss should equal -ln(1/V) = ln(V) (around 10.82 for V=50,257).",
            "codeExample": "import math\nprint('Expected initial loss for GPT-2:', math.log(50257))"
          }
        ],
        "concepts": [
          "Logits: raw positive or negative scores emitted by the final linear layer before normalization.",
          "Numerically stable Softmax: subtracting max(logits) before exp() so large numbers never overflow to NaN.",
          "Negative Log-Likelihood / Cross-Entropy Loss: -log(P(true_token))."
        ],
        "code": "import torch, torch.nn.functional as F\n\nlogits = torch.tensor([[2.5, 0.5, -1.0]])\ntarget = torch.tensor([0])\n\nprobs = F.softmax(logits, dim=-1)\nloss = F.cross_entropy(logits, target)\nprint(f\"Probs: {probs.tolist()} | Cross-Entropy Loss: {loss.item():.4f}\")",
        "quiz": {
          "question": "What input does PyTorch's F.cross_entropy(logits, targets) expect?",
          "options": [
            "Probabilities that already passed through F.softmax()",
            "Raw unnormalized logits directly from the linear LM Head",
            "Binary 0/1 integers only"
          ],
          "answerIndex": 1,
          "explanation": "F.cross_entropy applies log_softmax internally for numerical stability, so passing Softmax probabilities double-normalizes them."
        },
        "pitfall": "Passing Softmax probabilities into F.cross_entropy()! PyTorch's F.cross_entropy expects raw unnormalized logits.",
        "project": "Build a Softmax & Cross-Entropy visualizer notebook showing how initial random LLM loss equals ln(vocab_size).",
        "resourceTitle": "StatQuest — Cross-Entropy & Softmax Clearly Explained",
        "resourceNote": "Intuitive walkthrough of log-probabilities and why Cross-Entropy penalizes confident wrong predictions heavily.",
        "links": [
          {
            "type": "YouTube",
            "label": "StatQuest — Neural Networks Part 6: Cross Entropy",
            "url": "https://www.youtube.com/watch?v=6ArSys5qHAU"
          },
          {
            "type": "Community Guide",
            "label": "Victor Zhou — Softmax & Cross-Entropy Loss Explained Simply",
            "url": "https://victorzhou.com/blog/softmax/"
          }
        ]
      }
    ]
  },
  {
    "id": "phase-2",
    "number": "02",
    "category": "math-pytorch",
    "accent": "indigo",
    "duration": "Week 3–4",
    "title": "Neural Networks & Autograd from Scratch",
    "subtitle": "Demystify 'magic' training loops by building a scalar backpropagation engine (micrograd) before graduating to PyTorch nn.Module.",
    "plainBigPicture": "How does an AI actually 'learn' from its mistakes? Every calculation is recorded as a step-by-step recipe (a Computation Graph). When the model makes a wrong guess, 'Backpropagation' walks backward through that recipe using basic calculus to figure out which knobs ('Weights') to nudge slightly up or down.",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 2 Skills",
      "summary": "Every frontier model relies on reverse-mode automatic differentiation and gated MLP layers:",
      "examples": [
        "Llama 3, Mistral, and Gemma replaced older ReLU layers with the exact SwiGLU gated MLP architecture you code in Module 2.3.",
        "AdamW with decoupled weight decay (Module 2.4) trained GPT-3, Llama 3, and DeepSeek models across GPU clusters."
      ]
    },
    "jargonDecoder": [
      {
        "term": "Weights",
        "def": "Adjustable numbers inside the AI brain"
      },
      {
        "term": "Gradient",
        "def": "Direction to nudge a weight to reduce error"
      },
      {
        "term": "Backpropagation",
        "def": "Walking backward to assign blame for errors"
      },
      {
        "term": "Autograd",
        "def": "PyTorch's automatic gradient calculator"
      },
      {
        "term": "AdamW",
        "def": "Smart algorithm that updates the weights"
      }
    ],
    "achievements": [
      "Explain exactly what happens inside loss.backward() because you built a working scalar Autograd engine (micrograd) yourself.",
      "Construct multi-layer neural networks with modern non-linear activations (GELU & SwiGLU) and proper weight initialization.",
      "Write bug-free PyTorch training loops using AdamW, zero_grad(), train()/eval() switching, and activation diagnostics."
    ],
    "miniProject": "Code Micrograd from Scratch & Train a 2-Layer Neural Network Without PyTorch",
    "topics": [
      {
        "id": "p2-t1",
        "codeNum": "2.1",
        "animType": "autograd_graph",
        "title": "Computation Graphs & The Chain Rule",
        "duration": "2 Days",
        "plainEnglish": "Imagine a chain of gears: turning Gear A turns Gear B twice as fast, and Gear B turns Gear C three times as fast—so Gear C moves 2 x 3 = 6 times as fast as Gear A. That multiplication of step-by-step sensitivities is the 'Chain Rule'.",
        "liveExample": "PyTorch's dynamic autograd engine builds a DAG on every forward pass and traverses it backward via the Chain Rule.",
        "jargonTerms": [
          {
            "term": "Computation Graph",
            "def": "A flowchart showing every add/multiply step from input to final error."
          },
          {
            "term": "Derivative / Gradient",
            "def": "How much the final error changes if you nudge one number slightly."
          },
          {
            "term": "Chain Rule",
            "def": "Multiplying step-by-step sensitivities along a chain of operations."
          }
        ],
        "why": "Every neural network—from a 2-line equation to GPT-4—is a Directed Acyclic Graph (DAG) of elementary additions and multiplications differentiated via the Chain Rule.",
        "lessonSections": [
          {
            "heading": "1. Local Gradient vs. Upstream Gradient",
            "body": "Each node in a computation graph only needs to know its own local derivative. Multiplying upstream gradient (dL/dout) by local derivative (dout/din) passes blame backward.",
            "codeExample": "x, y, w, b = 2.0, 10.0, 3.0, 1.0\nz = w * x + b\nloss = (z - y) ** 2\ndloss_dz = 2 * (z - y)\ndloss_dw = dloss_dz * x"
          }
        ],
        "concepts": [
          "Derivatives as local sensitivity: measuring how nudging an input by h changes the final loss.",
          "Decomposing complex expressions into intermediate nodes.",
          "The Chain Rule: multiplying upstream gradient (dL/dout) by local derivative (dout/din)."
        ],
        "code": "x, y = 2.0, 10.0\nw, b = 3.0, 1.0\nz = w * x + b\ndiff = z - y\nloss = diff ** 2\n\ndloss_ddiff = 2 * diff\ndloss_dw = dloss_ddiff * x\nprint(\"Gradient wrt w:\", dloss_dw)",
        "quiz": {
          "question": "In L = (w * x + b - y)^2, if z = w * x + b and dL/dz = -6.0 while x = 2.0, what is dL/dw?",
          "options": [
            "-12.0",
            "-3.0",
            "+6.0"
          ],
          "answerIndex": 0,
          "explanation": "By the Chain Rule, dL/dw = (dL/dz) * (dz/dw) = -6.0 * x = -6.0 * 2.0 = -12.0."
        },
        "pitfall": "Overwriting gradients (=) instead of accumulating them (+=) when a variable is used in multiple branches of a graph!",
        "project": "Draw a 5-node computation graph on paper and verify your hand-calculated gradients against numerical finite differences.",
        "resourceTitle": "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)",
        "resourceNote": "Builds rock-solid geometric intuition for how gradients flow backward through expressions.",
        "links": [
          {
            "type": "YouTube",
            "label": "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)",
            "url": "https://www.youtube.com/watch?v=Ilg3gGewQ5U"
          },
          {
            "type": "Community Guide",
            "label": "Chris Olah's Blog — Calculus on Computational Graphs: Backpropagation",
            "url": "https://colah.github.io/posts/2015-08-Backprop/"
          }
        ]
      },
      {
        "id": "p2-t2",
        "codeNum": "2.2",
        "animType": "autograd_graph",
        "title": "Building micrograd: Scalar Autograd Engine",
        "duration": "4 Days",
        "plainEnglish": "When you call 'loss.backward()' in PyTorch, it feels like magic. Here you will build your own tiny 100-line clone of PyTorch's brain ('micrograd') using plain Python numbers.",
        "liveExample": "PyTorch's C++ libtorch Autograd engine is a tensorized version of this exact Value DAG + topological reverse traversal.",
        "jargonTerms": [
          {
            "term": "Autograd",
            "def": "Automatic differentiation—computing gradients automatically without hand-math."
          },
          {
            "term": "Topological Sort",
            "def": "Ordering graph steps so a node only runs backward after all nodes after it finish."
          }
        ],
        "why": "Writing Karpathy's ~100-line micrograd engine removes all mystery from PyTorch's autograd.",
        "lessonSections": [
          {
            "heading": "1. Wrapping Floats in a Differentiable Value Object",
            "body": "By overriding __add__ and __mul__ in Python, expressions like a * b + c automatically record their parent pointers and local backward closures.",
            "codeExample": "class Value:\n    def __init__(self, data, _children=()):\n        self.data, self.grad = float(data), 0.0\n        self._backward = lambda: None\n        self._prev = set(_children)"
          }
        ],
        "concepts": [
          "Wrapping floats in a Value class storing .data, .grad, ._prev children, and a ._backward() closure.",
          "Implementing __add__, __mul__, __pow__, tanh(), and relu() with local gradient rules.",
          "Topological sorting of the computation DAG so backward() visits every node in strict reverse dependency order."
        ],
        "code": "class Value:\n    def __init__(self, data, _children=(), _op=''):\n        self.data, self.grad = float(data), 0.0\n        self._backward = lambda: None\n        self._prev = set(_children)\n\n    def __add__(self, other):\n        other = other if isinstance(other, Value) else Value(other)\n        out = Value(self.data + other.data, (self, other), '+')\n        def _backward():\n            self.grad += 1.0 * out.grad\n            other.grad += 1.0 * out.grad\n        out._backward = _backward\n        return out",
        "quiz": {
          "question": "Why must backward() accumulate gradients with += instead of =?",
          "options": [
            "Because a single weight or variable can be used in multiple branches of the graph",
            "To make floating point numbers larger",
            "Because Python doesn't support = inside closures"
          ],
          "answerIndex": 0,
          "explanation": "Multivariable calculus states that if variable x feeds two paths, its total gradient is the sum of gradients from both paths."
        },
        "pitfall": "Forgetting to reset node.grad = 0.0 before each training iteration—accumulated gradients from previous steps will corrupt parameter updates.",
        "project": "Implement the complete Value autograd engine in Python and verify its gradients against PyTorch's torch.autograd.",
        "resourceTitle": "Andrej Karpathy — Building micrograd (Zero to Hero Ep. 1)",
        "resourceNote": "Code along line-by-line in JupyterLab; this is the single highest-ROI exercise for LLM beginners.",
        "links": [
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — The spelled-out intro to neural networks: building micrograd",
            "url": "https://www.youtube.com/watch?v=VMj-3S1tku0"
          },
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — Official micrograd GitHub Repository",
            "url": "https://github.com/karpathy/micrograd"
          }
        ]
      },
      {
        "id": "p2-t3",
        "codeNum": "2.3",
        "animType": "gradient_descent",
        "title": "Neurons, Layers, MLPs & Non-Linear Activations (ReLU, GELU, SwiGLU)",
        "duration": "3 Days",
        "plainEnglish": "A single artificial 'Neuron' multiplies inputs by weights, adds them up, and passes the result through a non-linear curve ('Activation Function' like GELU or SwiGLU) so the network can learn complex patterns.",
        "liveExample": "Around 2/3 of the 70 billion parameters in Llama-3 70B live inside its SwiGLU Feed-Forward MLP layers.",
        "jargonTerms": [
          {
            "term": "MLP",
            "def": "Multi-Layer Perceptron—a stack of linear layers with activation curves between them."
          },
          {
            "term": "Activation Function",
            "def": "A non-linear bend (ReLU, GELU, SwiGLU) allowing AI to learn complex rules."
          }
        ],
        "why": "Two-thirds of the parameters in a Transformer block live inside its Multi-Layer Perceptron (MLP), where non-linear activations unlock complex reasoning.",
        "lessonSections": [
          {
            "heading": "1. Why Non-Linearity Is Mandatory",
            "body": "Without an activation function between two linear layers (x @ W1) @ W2, matrix associativity collapses them into a single linear matrix x @ (W1 @ W2), incapable of learning non-linear functions.",
            "codeExample": "import torch.nn as nn, torch.nn.functional as F\n# SwiGLU gates an up-projection with a SiLU curve"
          }
        ],
        "concepts": [
          "Single Neuron -> Layer of Neurons -> Multi-Layer Perceptron (MLP) architecture.",
          "Why non-linearities are essential: stacking linear layers without activations collapses into a single matrix multiply.",
          "Modern LLM activations: ReLU vs. smooth GELU (GPT-2/3) vs. gated SwiGLU (Llama 3 / Gemma)."
        ],
        "code": "import torch, torch.nn as nn, torch.nn.functional as F\n\nclass SwiGLU(nn.Module):\n    def __init__(self, dim, hidden_dim):\n        super().__init__()\n        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)\n        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)\n        self.w_down = nn.Linear(hidden_dim, dim, bias=False)\n\n    def forward(self, x):\n        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))",
        "quiz": {
          "question": "What happens if you stack 10 nn.Linear layers with NO activation functions between them?",
          "options": [
            "It becomes 10x smarter than a Transformer",
            "Mathematically it collapses into a single linear matrix multiplication",
            "PyTorch raises a SyntaxError"
          ],
          "answerIndex": 1,
          "explanation": "Linear transformations composed together remain strictly linear; non-linear curves (GELU, SwiGLU) are required for universal function approximation."
        },
        "pitfall": "Dead ReLU neurons: if weights initialize poorly or learning rate is too high, negative pre-activations output 0 forever.",
        "project": "Build Neuron, Layer, and MLP classes on top of your micrograd Value class.",
        "resourceTitle": "StatQuest & 3Blue1Brown — Neural Networks & Activations",
        "resourceNote": "Visualizes how layered neurons compose edges and patterns into higher-level representations.",
        "links": [
          {
            "type": "YouTube",
            "label": "StatQuest — The Essential Main Ideas of Neural Networks",
            "url": "https://www.youtube.com/watch?v=CqOfi41LfDw"
          },
          {
            "type": "Community Guide",
            "label": "Papers With Code — SwiGLU & GELU Activation Functions Explained",
            "url": "https://paperswithcode.com/method/swiglu"
          }
        ]
      },
      {
        "id": "p2-t4",
        "codeNum": "2.4",
        "animType": "gradient_descent",
        "title": "Gradient Descent, Learning Rate & AdamW Optimizer",
        "duration": "3 Days",
        "plainEnglish": "Imagine standing blindfolded on a mountain and trying to reach the lowest valley by feeling the slope under your feet. 'AdamW' is a smart GPS optimizer that adapts step sizes individually for every single weight.",
        "liveExample": "Virtually every open-weights LLM (Llama, Mistral, Qwen, GPT-2) uses AdamW with betas=(0.9, 0.95) and decoupled weight decay.",
        "jargonTerms": [
          {
            "term": "Gradient Descent",
            "def": "Stepping opposite to the gradient to steadily shrink prediction error."
          },
          {
            "term": "Optimizer (AdamW)",
            "def": "Smart algorithm that adjusts step sizes individually for every single weight."
          }
        ],
        "why": "Plain SGD struggles on Transformers because embedding weights and attention projections have wildly different gradient scales. AdamW adapts step sizes per parameter.",
        "lessonSections": [
          {
            "heading": "1. The 3-Step PyTorch Update Cycle",
            "body": "Always clear stale gradients with optimizer.zero_grad(set_to_none=True), backpropagate loss.backward(), and apply updates with optimizer.step().",
            "codeExample": "optimizer.zero_grad(set_to_none=True)\nloss.backward()\noptimizer.step()"
          }
        ],
        "concepts": [
          "Vanilla Gradient Descent update rule: param = param - lr * param.grad.",
          "Momentum and RMSProp adaptive scaling.",
          "AdamW: combining adaptive per-parameter learning rates with decoupled weight decay regularization."
        ],
        "code": "import torch\n\nmodel = torch.nn.Linear(64, 64)\noptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)\nprint(\"AdamW initialized with lr=3e-4\")",
        "quiz": {
          "question": "Why is AdamW preferred over plain SGD for training Transformers?",
          "options": [
            "AdamW adapts step sizes per parameter using gradient momentum and variance",
            "SGD doesn't work on GPUs",
            "AdamW requires zero memory"
          ],
          "answerIndex": 0,
          "explanation": "Transformer embeddings and attention projections experience vastly different gradient magnitudes, which AdamW normalizes per weight."
        },
        "pitfall": "Using a huge learning rate like lr=0.1 with AdamW on a Transformer—LLM AdamW learning rates typically sit around 1e-4 to 6e-4.",
        "project": "Compare SGD, SGD+Momentum, and AdamW on a noisy 2D bowl in Python.",
        "resourceTitle": "StatQuest — Gradient Descent & Adam Optimizer Intuition",
        "resourceNote": "Clear visual explanation of why momentum and adaptive learning rates speed up convergence.",
        "links": [
          {
            "type": "YouTube",
            "label": "StatQuest — Gradient Descent, Step-by-Step",
            "url": "https://www.youtube.com/watch?v=sDv4f4s2SB8"
          },
          {
            "type": "Community Guide",
            "label": "Distill.pub — Why Momentum Really Works (Interactive Visual Essay)",
            "url": "https://distill.pub/2017/momentum/"
          }
        ]
      },
      {
        "id": "p2-t5",
        "codeNum": "2.5",
        "animType": "gradient_descent",
        "title": "Training Loop Anatomy, Initialization & Vanishing/Exploding Gradients",
        "duration": "3 Days",
        "plainEnglish": "Before training starts, all weights are filled with tiny random numbers ('Initialization'). A clean 'Training Loop' repeats guess -> grade -> nudge thousands of times safely.",
        "liveExample": "GPT-2 and GPT-3 scale residual output projection initialization by 1 / sqrt(2 * n_layers) so deep 96-layer stacks don't explode at step 0.",
        "jargonTerms": [
          {
            "term": "Initialization",
            "def": "Setting smart starting random numbers in weights before step 1."
          },
          {
            "term": "model.eval()",
            "def": "Switches off training-only behaviors (like Dropout) when testing/chatting."
          }
        ],
        "why": "If initial weights are too large, activations explode to NaN on step 1; if too small, gradients vanish to zero.",
        "lessonSections": [
          {
            "heading": "1. Switching Between train() and eval() Modes",
            "body": "Dropout layers randomly zero out activations during training to prevent overfitting, so you must call model.eval() before generating text.",
            "codeExample": "model.eval()\nwith torch.no_grad():\n    logits = model(tokens)"
          }
        ],
        "concepts": [
          "The canonical PyTorch training loop: forward -> compute loss -> zero_grad -> backward -> clip_grad_norm -> step.",
          "GPT-style scaled normal initialization (std = 0.02).",
          "Diagnosing activation saturation and gradient norms across layers."
        ],
        "code": "import torch.nn as nn\n\ndef init_weights(module):\n    if isinstance(module, nn.Linear):\n        nn.init.normal_(module.weight, mean=0.0, std=0.02)\n        if module.bias is not None:\n            nn.init.zeros_(module.bias)",
        "quiz": {
          "question": "What happens if you forget to call model.eval() when generating text from a model trained with Dropout?",
          "options": [
            "Dropout remains active and randomly zeroes out neurons during generation, degrading output quality",
            "The model runs twice as fast",
            "PyTorch refuses to run forward()"
          ],
          "answerIndex": 0,
          "explanation": "model.eval() disables Dropout randomness so inference is clean and deterministic."
        },
        "pitfall": "Forgetting model.eval() during validation/inference and model.train() during training.",
        "project": "Write a clean, reusable PyTorch trainer function with train/eval mode switching and weight initialization.",
        "resourceTitle": "Andrej Karpathy — Building makemore Part 3: Activations & Gradients",
        "resourceNote": "Deep diagnostic masterclass on inspecting activation histograms and gradient flow.",
        "links": [
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — Building makemore Part 3: Activations & Gradients",
            "url": "https://www.youtube.com/watch?v=P6sfmUTpUmc"
          },
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy Blog — A Recipe for Training Neural Networks",
            "url": "https://karpathy.github.io/2019/04/25/recipe/"
          }
        ]
      }
    ]
  },
  {
    "id": "phase-3",
    "number": "03",
    "category": "math-pytorch",
    "accent": "teal",
    "duration": "Week 5",
    "title": "Text Tokenization & Data Pipelines",
    "subtitle": "Convert raw text corpora into subword token IDs using Byte-Pair Encoding (BPE) and construct sliding-window PyTorch DataLoaders.",
    "plainBigPicture": "Neural networks can only do math on numbers, never raw English letters. A 'Tokenizer' acts like a secret decoder ring: it chops text into common chunks ('Tokens') and assigns each chunk an integer ID.",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 3 Skills",
      "summary": "Tokenization directly explains famous quirks in ChatGPT and Claude:",
      "examples": [
        "Why LLMs struggle to count how many 'r's are in 'strawberry': BPE compresses 'straw' and 'berry' into single integer IDs so the model never sees individual letters!",
        "GPT-4o expanded its BPE vocabulary from 50k to 200k tokens (o200k_base) to make multilingual & code generation 2x faster."
      ]
    },
    "jargonDecoder": [
      {
        "term": "Token",
        "def": "A word or common word-piece (e.g., 'ing')"
      },
      {
        "term": "Tokenizer",
        "def": "Translates text <-> integer token IDs"
      },
      {
        "term": "BPE",
        "def": "Compression algorithm that merges frequent letter pairs"
      },
      {
        "term": "Context Window",
        "def": "Max tokens the model can read at once"
      },
      {
        "term": "DataLoader",
        "def": "Feeds shuffled training batches to the GPU"
      }
    ],
    "achievements": [
      "Code a working Byte-Pair Encoding (BPE) tokenizer from scratch in pure Python (train, encode, and decode).",
      "Handle UTF-8 byte streams and special control tokens (<|endoftext|>, <|user|>, <|assistant|>) without crashes.",
      "Build a PyTorch sliding-window Dataset & DataLoader producing (X, Y) next-token training pairs."
    ],
    "miniProject": "Code a Complete Byte-Pair Encoding (BPE) Tokenizer & Sliding-Window DataLoader",
    "topics": [
      {
        "id": "p3-t1",
        "codeNum": "3.1",
        "animType": "bpe_merge",
        "title": "Character-Level vs. Word-Level vs. Subword Tokenization",
        "duration": "2 Days",
        "plainEnglish": "Subword Tokenization splits rare words into reusable roots like 'un-' + 'believ' + '-able' while keeping frequent words intact as single IDs.",
        "liveExample": "OpenAI's tiktoken and Meta's SentencePiece both use subword tokenization averaging ~0.75 English words per token.",
        "jargonTerms": [
          {
            "term": "Subword Token",
            "def": "1 token = frequent word or root/suffix chunk (~0.75 words on average)."
          }
        ],
        "why": "Tokenization defines the atomic symbols your LLM sees and predicts—directly governing context length efficiency, spelling, and code accuracy.",
        "lessonSections": [
          {
            "heading": "1. Why Subword Beats Character and Whole-Word",
            "body": "Character-level makes sequences 5x longer (quadratic attention cost), while whole-word fails on typos and code identifiers. Subword BPE balances vocabulary size (~50k–128k) with compact sequences.",
            "codeExample": "# 'unbelievable' -> ['un', 'believ', 'able'] (3 tokens instead of 12 chars)"
          }
        ],
        "concepts": [
          "Character-level vs. Word-level vs. Subword tokenization (BPE / WordPiece / Unigram).",
          "Leading spaces in GPT tokenizers ('apple' vs ' apple')."
        ],
        "code": "# \"unbelievable\" -> [\"un\", \"believ\", \"able\"] -> [403, 18924, 540]",
        "quiz": {
          "question": "Why do modern LLMs use subword tokenization instead of pure character-level tokenization?",
          "options": [
            "Character-level makes token sequences ~4-5x longer, dramatically increasing attention compute cost",
            "Characters cannot be represented as integers",
            "Subword tokenization eliminates the need for embeddings"
          ],
          "answerIndex": 0,
          "explanation": "Self-attention scales quadratically with sequence length T, so compressing 12 characters into 3 subwords speeds up training and inference massively."
        },
        "pitfall": "Leading spaces matter! In GPT tokenizers, 'apple' and ' apple' have completely different token IDs.",
        "project": "Compare sequence lengths across character-level, word-level, and subword tokenization.",
        "resourceTitle": "Hugging Face LLM Course — Tokenizers & Karpathy's GPT Tokenizer",
        "resourceNote": "Shows concrete examples of how tokenization quirks explain LLM failure modes.",
        "links": [
          {
            "type": "Community Guide",
            "label": "OpenAI Tiktokenizer — Interactive Web Token Splitter",
            "url": "https://tiktokenizer.vercel.app/"
          },
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — Let's Build the GPT Tokenizer (Full Lecture)",
            "url": "https://www.youtube.com/watch?v=zduSFxRajkE"
          }
        ]
      },
      {
        "id": "p3-t2",
        "codeNum": "3.2",
        "animType": "bpe_merge",
        "title": "Unicode, UTF-8 Byte Encoding & Vocabulary Trade-offs",
        "duration": "2 Days",
        "plainEnglish": "UTF-8 breaks any letter, Tamil/Hindi script, or emoji into 1 to 4 raw bytes (integers 0–255). Starting our tokenizer with those 256 base bytes guarantees 100% coverage of every language.",
        "liveExample": "GPT-2, GPT-4, and Llama 3 all use Byte-Level BPE so they never emit <UNK> tokens on emojis or code.",
        "jargonTerms": [
          {
            "term": "UTF-8 Bytes",
            "def": "Encoding any Unicode character into 1–4 byte integers (0 to 255)."
          }
        ],
        "why": "Modern tokenizers operate on raw UTF-8 bytes (0–255) so every emoji, symbol, and language is representable without <UNK> tokens.",
        "lessonSections": [
          {
            "heading": "1. Converting Strings to Base Byte IDs",
            "body": "In Python, list(text.encode('utf-8')) immediately converts any string into integers between 0 and 255.",
            "codeExample": "raw_bytes = list('Hello 🚀'.encode('utf-8'))\nprint(raw_bytes)"
          }
        ],
        "concepts": [
          "Unicode code points vs. UTF-8 variable-length byte encoding (1 to 4 bytes).",
          "Base vocabulary of 256 raw bytes guarantees 100% coverage."
        ],
        "code": "text = \"Hello 🚀 LLM\"\nraw_bytes = list(text.encode(\"utf-8\"))\nprint(\"UTF-8 byte IDs:\", raw_bytes)\nprint(\"Decoded back:\", bytes(raw_bytes).decode(\"utf-8\"))",
        "quiz": {
          "question": "How many base tokens does a Byte-Level BPE tokenizer start with before any merges?",
          "options": [
            "26 (A-Z)",
            "256 (all possible byte values 0..255)",
            "50,257"
          ],
          "answerIndex": 1,
          "explanation": "Any UTF-8 string is a sequence of bytes 0..255, giving a lossless 256-token base alphabet."
        },
        "pitfall": "Always use .decode('utf-8', errors='replace') when decoding arbitrary generated IDs to avoid UnicodeDecodeError crashes.",
        "project": "Write a helper script that inspects UTF-8 byte lengths across English, Tamil/Hindi, Python code, and emojis.",
        "resourceTitle": "Andrej Karpathy — Let's Build the GPT Tokenizer (Unicode & UTF-8 Section)",
        "resourceNote": "Explains why starting from UTF-8 bytes eliminates out-of-vocabulary errors forever.",
        "links": [
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — Let's Build the GPT Tokenizer",
            "url": "https://www.youtube.com/watch?v=zduSFxRajkE"
          },
          {
            "type": "Blog / Docs",
            "label": "Hugging Face LLM Course — Byte-Pair Encoding Tokenization",
            "url": "https://huggingface.co/learn/llm-course/chapter6/5"
          }
        ]
      },
      {
        "id": "p3-t3",
        "codeNum": "3.3",
        "animType": "bpe_merge",
        "title": "Coding Byte-Pair Encoding (BPE) from Scratch (minbpe)",
        "duration": "3 Days",
        "plainEnglish": "BPE scans a corpus, finds the two adjacent tokens appearing most frequently, mints a brand-new token ID for that pair, and repeats until reaching the target vocabulary size.",
        "liveExample": "Karpathy's minbpe replicates OpenAI's exact GPT-4 tokenizer algorithm in ~100 lines of readable Python.",
        "jargonTerms": [
          {
            "term": "BPE (Byte-Pair Encoding)",
            "def": "Iteratively merging the most frequent adjacent token pair into one new token."
          },
          {
            "term": "Merge Table",
            "def": "The ordered rulebook of pair merges learned during tokenizer training."
          }
        ],
        "why": "BPE is the exact algorithm used to train GPT-2, GPT-4, and Llama tokenizers.",
        "lessonSections": [
          {
            "heading": "1. The Pair-Merging Loop",
            "body": "Each merge step replaces every occurrence of the top pair (p0, p1) with a new integer ID (256, 257, ...).",
            "codeExample": "def merge_pair(ids, pair, new_id):\n    out, i = [], 0\n    while i < len(ids):\n        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n            out.append(new_id); i += 2\n        else:\n            out.append(ids[i]); i += 1\n    return out"
          }
        ],
        "concepts": [
          "Counting all adjacent token pairs across the corpus with a frequency dictionary.",
          "Merging the highest-frequency pair into a new token ID and storing the merge rule.",
          "Encoding new text by applying learned merges in strict priority order."
        ],
        "code": "def merge_pair(ids, pair, new_id):\n    out, i = [], 0\n    while i < len(ids):\n        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n            out.append(new_id)\n            i += 2\n        else:\n            out.append(ids[i])\n            i += 1\n    return out",
        "quiz": {
          "question": "When encoding new text with a trained BPE tokenizer, in what order must merges be applied?",
          "options": [
            "Alphabetical order",
            "In the exact priority order (lowest merge rank first) in which they were learned during training",
            "Random order"
          ],
          "answerIndex": 1,
          "explanation": "Earlier merges form sub-components required by later merges, so lowest-index merges must execute first."
        },
        "pitfall": "In encode(), always apply the merge with the lowest merge index at each step—not arbitrary dictionary iteration order!",
        "project": "Build a BasicTokenizer class with train(), encode(), and decode() and train a 512-token vocab on TinyShakespeare.",
        "resourceTitle": "Andrej Karpathy — minbpe Repository & Tokenizer Lecture",
        "resourceNote": "Walks through BasicTokenizer and RegexTokenizer step by step.",
        "links": [
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — Let's Build the GPT Tokenizer",
            "url": "https://www.youtube.com/watch?v=zduSFxRajkE"
          },
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — Official minbpe GitHub Repository",
            "url": "https://github.com/karpathy/minbpe"
          }
        ]
      },
      {
        "id": "p3-t4",
        "codeNum": "3.4",
        "animType": "bpe_merge",
        "title": "Special Tokens (<|endoftext|>, PAD, BOS/EOS) & Context Windows",
        "duration": "2 Days",
        "plainEnglish": "Special control tokens like <|endoftext|> and <|assistant|> act as structural delimiters so the model knows where documents end and whose turn it is in a chat.",
        "liveExample": "ChatGPT's ChatML format uses <|im_start|>system, <|im_start|>user, and <|im_start|>assistant special token IDs.",
        "jargonTerms": [
          {
            "term": "Special Tokens",
            "def": "Reserved control IDs like <|endoftext|> never split by BPE."
          },
          {
            "term": "EOS",
            "def": "End-of-Sequence stop token telling the model to stop generating."
          }
        ],
        "why": "Special control tokens delimit separate documents during pretraining and separate User vs. Assistant turns during chat fine-tuning.",
        "lessonSections": [
          {
            "heading": "1. ChatML Conversation Delimiters",
            "body": "Special tokens are matched atomically before regex/BPE splitting runs so they always map to a single dedicated integer ID.",
            "codeExample": "special_tokens = {'<|endoftext|>': 50256, '<|user|>': 50257, '<|assistant|>': 50258}"
          }
        ],
        "concepts": [
          "Document separators (<|endoftext|>) preventing unrelated articles from bleeding together.",
          "Conversation delimiters (<|im_start|>, <|im_end|>) and padding tokens.",
          "Context length (block_size / max_seq_len)."
        ],
        "code": "special_tokens = {\n    \"<|endoftext|>\": 50256,\n    \"<|user|>\": 50257,\n    \"<|assistant|>\": 50258\n}",
        "quiz": {
          "question": "How does an autoregressive LLM know when to stop generating a chat reply?",
          "options": [
            "It always generates exactly 500 words",
            "It predicts the special End-of-Sequence (<|endoftext|> / <|im_end|>) token ID",
            "It stops at the first period"
          ],
          "answerIndex": 1,
          "explanation": "During SFT, assistant replies end with an EOS token; when sampled during inference, the generation loop halts."
        },
        "pitfall": "Allowing regular BPE rules to split '<|endoftext|>' into separate characters.",
        "project": "Extend your BPE tokenizer to register and encode/decode custom special tokens.",
        "resourceTitle": "Sebastian Raschka — Chapter 2: Working with Text Data & Special Tokens",
        "resourceNote": "Shows practical examples of handling unknown words and document boundary markers.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 2 Code Notebooks",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02"
          },
          {
            "type": "Blog / Docs",
            "label": "Hugging Face Docs — Chat Templates & Special Tokens",
            "url": "https://huggingface.co/docs/transformers/main/en/chat_templating"
          }
        ]
      },
      {
        "id": "p3-t5",
        "codeNum": "3.5",
        "animType": "sliding_window",
        "title": "Sliding-Window Input/Target Pairs (X, Y) & PyTorch Dataset/DataLoader",
        "duration": "2 Days",
        "plainEnglish": "LLMs learn via self-supervised fill-in-the-blank: Input X = [10, 20, 30] and Target Y = [20, 30, 40] (shifted 1 token right).",
        "liveExample": "Pretraining pipelines pack trillions of web tokens into fixed-length blocks where every position predicts the next token simultaneously.",
        "jargonTerms": [
          {
            "term": "Input / Target (X, Y)",
            "def": "Y is simply X shifted forward by 1 token (the 'next token' answer key)."
          },
          {
            "term": "DataLoader",
            "def": "PyTorch utility that packages samples and serves shuffled batches."
          }
        ],
        "why": "A single sequence of T+1 tokens yields T self-supervised next-token prediction training examples in parallel.",
        "lessonSections": [
          {
            "heading": "1. Shifting Targets by One Position",
            "body": "For any sequence slice of length T+1, x = chunk[:-1] and y = chunk[1:] creates T parallel next-token training targets.",
            "codeExample": "x = torch.tensor(token_ids[i : i + max_len])\ny = torch.tensor(token_ids[i + 1 : i + max_len + 1])"
          }
        ],
        "concepts": [
          "Shifting targets by 1 index: X = data[i : i+T], Y = data[i+1 : i+T+1].",
          "Stride parameter: overlapping vs. non-overlapping chunks.",
          "Wrapping token arrays in torch.utils.data.Dataset and DataLoader."
        ],
        "code": "import torch\nfrom torch.utils.data import Dataset\n\nclass GPTDataset(Dataset):\n    def __init__(self, token_ids, max_len, stride):\n        self.inputs, self.targets = [], []\n        for i in range(0, len(token_ids) - max_len, stride):\n            self.inputs.append(torch.tensor(token_ids[i : i + max_len]))\n            self.targets.append(torch.tensor(token_ids[i + 1 : i + max_len + 1]))\n    def __len__(self): return len(self.inputs)\n    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]",
        "quiz": {
          "question": "If Input X is [464, 3139, 286, 4881], what is the matching next-token Target Y?",
          "options": [
            "[464, 3139, 286, 4881]",
            "[3139, 286, 4881, 318]",
            "[4881, 286, 3139, 464]"
          ],
          "answerIndex": 1,
          "explanation": "Each position t in X predicts token t+1, so Y is shifted one index to the right."
        },
        "pitfall": "Setting stride=1 on a large corpus creates near-duplicate overlapping windows that cause rapid memorization.",
        "project": "Create a working GPTDataset and DataLoader over TinyShakespeare and verify batch shapes X: (32, 128) and Y: (32, 128).",
        "resourceTitle": "PyTorch Datasets & DataLoaders + Sebastian Raschka Ch. 2.6",
        "resourceNote": "Includes clear diagrams of input-target shifting and DataLoader batching.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Sliding Window DataLoader Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02"
          },
          {
            "type": "Blog / Docs",
            "label": "PyTorch Official Tutorials — Datasets & DataLoaders",
            "url": "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html"
          }
        ]
      }
    ]
  },
  {
    "id": "phase-4",
    "number": "04",
    "category": "transformer",
    "accent": "indigo",
    "duration": "Week 6–8",
    "title": "Transformer Architecture & Self-Attention (The Heart of GPT)",
    "subtitle": "Build every internal component of a Decoder-Only Transformer from scratch in pure PyTorch—from embeddings and QKV causal attention to stacked GPT blocks.",
    "plainBigPicture": "This is the core engine behind ChatGPT, Claude, and Llama. First, token IDs become rich meaning vectors ('Embeddings'). Next, 'Self-Attention' lets words look back at earlier words for context. Stacking these blocks creates a GPT!",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 4 Skills",
      "summary": "Every frontier model (GPT-4, Claude 3.5, Llama 3, DeepSeek) is built out of these exact Decoder-Only Transformer blocks:",
      "examples": [
        "RoPE positional rotations (Module 4.1) enabled Llama 3 & Claude to scale context windows from 2K tokens up to 128K–200K tokens.",
        "Causal lower-triangular masking (Module 4.3) allows streaming token-by-token chat via KV Caching without recomputing past tokens."
      ]
    },
    "jargonDecoder": [
      {
        "term": "Embedding",
        "def": "List of numbers capturing a word's meaning"
      },
      {
        "term": "Self-Attention",
        "def": "Words looking back at earlier context words"
      },
      {
        "term": "Q, K, V",
        "def": "Query (question), Key (label), Value (info)"
      },
      {
        "term": "Causal Mask",
        "def": "Blinder hiding future words during training"
      },
      {
        "term": "LayerNorm",
        "def": "Keeps numbers balanced across deep layers"
      }
    ],
    "achievements": [
      "Implement Scaled Dot-Product Self-Attention and Multi-Head Attention with causal lower-triangular masking from scratch.",
      "Combine Token Embeddings, Positional Encodings (Learned & RoPE), LayerNorm, Residual Connections, and Feed-Forward MLPs.",
      "Assemble a complete, parameter-verified Decoder-Only GPT model in pure PyTorch ready for pretraining."
    ],
    "miniProject": "Assemble a Complete Decoder-Only GPT Model in Pure PyTorch & Verify Forward Pass",
    "topics": [
      {
        "id": "p4-t1",
        "codeNum": "4.1",
        "animType": "embeddings_pos",
        "title": "Token Embeddings & Positional Encodings (Learned vs. Sinusoidal vs. RoPE)",
        "duration": "3 Days",
        "plainEnglish": "An 'Embedding' translates each token ID into a learned list of 256+ numbers where similar concepts sit close together, plus a 'Positional Encoding' stamp so word order matters.",
        "liveExample": "Llama 3, Mistral, and Qwen use Rotary Position Embeddings (RoPE) so relative distance between words is encoded directly in Q and K dot products.",
        "jargonTerms": [
          {
            "term": "nn.Embedding",
            "def": "A lookup table translating integer token IDs into rich decimal vectors."
          },
          {
            "term": "RoPE",
            "def": "Rotary Position Embedding—rotates vectors by position angle."
          }
        ],
        "why": "Self-attention treats inputs as an unordered set; positional encodings inject word-order geometry.",
        "lessonSections": [
          {
            "heading": "1. Combining Semantic + Positional Vectors",
            "body": "In GPT-2, token embeddings and position embeddings share the exact same dimension emb_dim and are added element-wise.",
            "codeExample": "x = tok_emb(idx) + pos_emb(torch.arange(T))"
          }
        ],
        "concepts": [
          "nn.Embedding(vocab_size, emb_dim): differentiable lookup table.",
          "Absolute Learned Positional Embeddings (GPT-2) vs. Sinusoidal vs. RoPE."
        ],
        "code": "import torch, torch.nn as nn\n\nB, T, vocab_size, emb_dim, max_len = 2, 16, 50257, 256, 512\ntok_emb = nn.Embedding(vocab_size, emb_dim)\npos_emb = nn.Embedding(max_len, emb_dim)\nidx = torch.randint(0, vocab_size, (B, T))\nx = tok_emb(idx) + pos_emb(torch.arange(T))\nprint(\"Input embedding tensor shape:\", x.shape)",
        "quiz": {
          "question": "Why do Transformers require Positional Encodings?",
          "options": [
            "Because self-attention is permutation-equivariant and has no built-in notion of token order",
            "To compress vocabulary size",
            "To replace Softmax"
          ],
          "answerIndex": 0,
          "explanation": "Without positional information, 'dog bites man' and 'man bites dog' produce identical attention sets."
        },
        "pitfall": "Passing sequences longer than max_len into a learned positional embedding table.",
        "project": "Plot the cosine similarity matrix of a learned positional embedding table before and after training.",
        "resourceTitle": "Sebastian Raschka — Positional Embeddings Blog & Jay Alammar GPT-2",
        "resourceNote": "Compare additive GPT-2 positional embeddings with modern multiplicative RoPE rotations.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Jay Alammar — The Illustrated GPT-2 (Embeddings & Decoder)",
            "url": "https://jalammar.github.io/illustrated-gpt2/"
          },
          {
            "type": "Community Guide",
            "label": "EleutherAI Blog — Rotary Embeddings (RoPE)",
            "url": "https://blog.eleuther.ai/rotary-embeddings/"
          }
        ]
      },
      {
        "id": "p4-t2",
        "codeNum": "4.2",
        "animType": "qkv_attention",
        "title": "Query, Key, Value (QKV) Projections & Scaled Dot-Product Attention",
        "duration": "4 Days",
        "plainEnglish": "Every word emits a 'Query' (what I am looking for), a 'Key' (what label I advertise), and a 'Value' (the actual information I share if matched).",
        "liveExample": "When you ask ChatGPT 'The trophy didn't fit in the suitcase because it was too small—what was small?', attention heads route 'it' strongly back to 'suitcase'.",
        "jargonTerms": [
          {
            "term": "Query (Q)",
            "def": "Vector representing what context a token is currently searching for."
          },
          {
            "term": "Key (K)",
            "def": "Vector advertising what grammatical/semantic role a token holds."
          },
          {
            "term": "Value (V)",
            "def": "The actual content vector passed forward when Q matches K."
          }
        ],
        "why": "Self-Attention lets each token dynamically query earlier tokens for context.",
        "lessonSections": [
          {
            "heading": "1. Why We Divide by sqrt(d_k)",
            "body": "For large head dimensions d_k, dot products grow large in variance, pushing Softmax into near-zero gradients. Scaling by 1/sqrt(d_k) keeps variance at 1.",
            "codeExample": "scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)"
          }
        ],
        "concepts": [
          "Linear projections W_q, W_k, W_v.",
          "Raw affinity scores: Q @ K.transpose(-2, -1) / sqrt(d_k).",
          "Weighted sum of Value vectors via Softmax."
        ],
        "code": "import torch, torch.nn.functional as F\n\nB, T, d_k = 2, 6, 32\nQ, K, V = torch.randn(B, T, d_k), torch.randn(B, T, d_k), torch.randn(B, T, d_k)\nscores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)\nattn_weights = F.softmax(scores, dim=-1)\ncontext = attn_weights @ V\nprint(\"Context shape:\", context.shape)",
        "quiz": {
          "question": "Why do we divide Q @ K^T by sqrt(d_k) before applying Softmax?",
          "options": [
            "To prevent large dot products from pushing Softmax into saturated regions with vanishing gradients",
            "To double the sequence length",
            "To convert float32 into int64"
          ],
          "answerIndex": 0,
          "explanation": "Scaling by 1/sqrt(d_k) normalizes dot-product variance back to ~1 regardless of head dimension."
        },
        "pitfall": "Forgetting to divide by sqrt(d_k)!",
        "project": "Implement a single-head SelfAttention class in PyTorch and verify that every row of attn_weights sums to 1.0.",
        "resourceTitle": "3Blue1Brown — Attention in Transformers, Visually Explained",
        "resourceNote": "Best geometric animation on the internet for understanding Q, K, V dot products.",
        "links": [
          {
            "type": "YouTube",
            "label": "3Blue1Brown — Attention in Transformers, Step-by-Step (Ch. 6)",
            "url": "https://www.youtube.com/watch?v=eMlx5fFNoYc"
          },
          {
            "type": "Community Guide",
            "label": "Harvard NLP — The Annotated Transformer",
            "url": "https://nlp.seas.harvard.edu/annotated-transformer/"
          }
        ]
      },
      {
        "id": "p4-t3",
        "codeNum": "4.3",
        "animType": "causal_mask",
        "title": "Causal Masking (tril) for Autoregressive Next-Token Prediction",
        "duration": "2 Days",
        "plainEnglish": "Without a 'Causal Mask', word #3 could cheat by peeking ahead at word #4 during parallel training! We fill the upper-right triangle with -inf so future words get 0% attention.",
        "liveExample": "Decoder-only LLMs (GPT, Claude, Llama) use causal masking so all T token positions train simultaneously without future leakage.",
        "jargonTerms": [
          {
            "term": "Causal Mask",
            "def": "Triangular barrier blocking tokens from peeking at future tokens."
          },
          {
            "term": "torch.tril",
            "def": "'Triangle Lower'—keeps the bottom-left triangle and zeroes out top-right."
          }
        ],
        "why": "Causal masking prevents token t from peeking at future tokens t+1..T so parallel training matches sequential inference.",
        "lessonSections": [
          {
            "heading": "1. Masking with -inf Before Softmax",
            "body": "Because exp(-inf) = 0, masking upper-triangle scores with float('-inf') before Softmax guarantees exact 0.0 weights on future positions while keeping row sums at 1.0.",
            "codeExample": "masked = scores.masked_fill(mask == 0, float('-inf'))\nprobs = torch.softmax(masked, dim=-1)"
          }
        ],
        "concepts": [
          "Lower-triangular matrix torch.tril(torch.ones(T, T)).",
          "Masking future positions with -inf BEFORE Softmax.",
          "Registering the mask via self.register_buffer('mask', ...)."
        ],
        "code": "import torch\nT = 4\nscores = torch.randn(T, T)\nmask = torch.tril(torch.ones(T, T))\nmasked_scores = scores.masked_fill(mask == 0, float('-inf'))\nprint(torch.softmax(masked_scores, dim=-1))",
        "quiz": {
          "question": "Why must future attention scores be set to -inf BEFORE Softmax rather than 0 AFTER Softmax?",
          "options": [
            "Because exp(-inf) = 0, ensuring future tokens get 0% weight AND the remaining past weights still sum cleanly to 1.0",
            "Because PyTorch doesn't allow 0 inside tensors",
            "To make the matrix symmetric"
          ],
          "answerIndex": 0,
          "explanation": "Zeroing probabilities after Softmax would leave rows summing to less than 1.0, distorting activations."
        },
        "pitfall": "Masking with 0 AFTER Softmax instead of -inf BEFORE Softmax!",
        "project": "Add causal masking with register_buffer to your SelfAttention module.",
        "resourceTitle": "Andrej Karpathy — Let's Build GPT (The Mathematical Trick in Self-Attention)",
        "resourceNote": "Walks through weighted aggregation of past tokens using lower-triangular matrices.",
        "links": [
          {
            "type": "YouTube",
            "label": "Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out",
            "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY"
          },
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Understanding & Coding Self-Attention From Scratch",
            "url": "https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html"
          }
        ]
      },
      {
        "id": "p4-t4",
        "codeNum": "4.4",
        "animType": "qkv_attention",
        "title": "Multi-Head Self-Attention & Tensor Reshaping / Concatenation",
        "duration": "4 Days",
        "plainEnglish": "'Multi-Head Attention' splits the embedding into 6 to 32 parallel spotlights ('Heads') so different heads can track grammar, pronouns, and punctuation simultaneously.",
        "liveExample": "GPT-3 uses 96 parallel attention heads per layer; modern models like Llama-3 use Grouped-Query Attention (GQA) to share K/V heads for faster inference.",
        "jargonTerms": [
          {
            "term": "Attention Head",
            "def": "An independent QKV spotlight focusing on one type of word relationship."
          },
          {
            "term": ".contiguous()",
            "def": "Re-aligns tensor memory layout after swapping axes so .view() works."
          }
        ],
        "why": "Multiple heads let the model attend simultaneously to syntactic structure, entity references, and local phrasing across distinct subspaces.",
        "lessonSections": [
          {
            "heading": "1. Batched 4D Head Splitting",
            "body": "Reshaping (B, T, C) -> (B, T, n_heads, head_dim) and transposing to (B, n_heads, T, head_dim) executes all attention heads in a single batched GPU kernel.",
            "codeExample": "q = q.view(B, T, n_heads, head_dim).transpose(1, 2)"
          }
        ],
        "concepts": [
          "Splitting C = n_heads * head_dim.",
          "Batched 4D tensor reshape and transpose.",
          "Merging heads back via .transpose(1, 2).contiguous().view(B, T, C)."
        ],
        "code": "import torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, emb_dim, n_heads):\n        super().__init__()\n        self.n_heads, self.head_dim = n_heads, emb_dim // n_heads\n        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)\n        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)\n\n    def forward(self, x):\n        B, T, C = x.shape\n        q, k, v = self.qkv(x).chunk(3, dim=-1)\n        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n        return self.proj(q.transpose(1, 2).contiguous().view(B, T, C))",
        "quiz": {
          "question": "Why is .contiguous() required before .view(B, T, C) after calling .transpose(1, 2)?",
          "options": [
            "Because .transpose() alters tensor strides without moving memory, and .view() requires contiguous memory",
            "To move the tensor to CPU",
            "To apply LayerNorm automatically"
          ],
          "answerIndex": 0,
          "explanation": "Transposing creates a non-contiguous view; calling .contiguous() packs memory linearly so .view() can merge head dimensions."
        },
        "pitfall": "Forgetting .contiguous() before .view(B, T, C) after .transpose(1, 2).",
        "project": "Code a complete batched MultiHeadAttention module in PyTorch.",
        "resourceTitle": "Sebastian Raschka — Multi-Head Attention & Vizuara SLM Walkthrough",
        "resourceNote": "Compares naive stacked heads with efficient batched weight splits.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 3.5 Multi-Head Attention Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03"
          },
          {
            "type": "YouTube",
            "label": "Vizuara — Build a Small Language Model (SLM) From Scratch",
            "url": "https://www.youtube.com/watch?v=pOFcwcwtv3k"
          }
        ]
      },
      {
        "id": "p4-t5",
        "codeNum": "4.5",
        "animType": "transformer_block",
        "title": "LayerNorm / RMSNorm, Residual Connections & Feed-Forward MLP Blocks",
        "duration": "3 Days",
        "plainEnglish": "'LayerNorm' re-centers every token's numbers to a clean bell curve, and 'Residual Connections' (x = x + layer(x)) act like an express elevator carrying gradients cleanly top-to-bottom.",
        "liveExample": "Llama 3 and Gemma use Pre-Norm with RMSNorm—simplifying LayerNorm by dropping mean subtraction for a 10–15% speedup.",
        "jargonTerms": [
          {
            "term": "LayerNorm / RMSNorm",
            "def": "Re-scales each token's numbers to mean 0, std 1 so deep layers stay stable."
          },
          {
            "term": "Residual Connection",
            "def": "Adding the input directly to the output (x + block(x)) as a gradient highway."
          }
        ],
        "why": "Deep Transformers only train stably because residual highways let gradients flow unimpeded from loss straight back to early embeddings.",
        "lessonSections": [
          {
            "heading": "1. Pre-Norm Residual Stream",
            "body": "Modern Transformers normalize BEFORE attention and MLP (x + Attn(Norm(x))), keeping an unobstructed identity path from loss to embeddings.",
            "codeExample": "x = x + self.attn(self.ln1(x))\nx = x + self.mlp(self.ln2(x))"
          }
        ],
        "concepts": [
          "Pre-Norm Transformer Block: x = x + Attention(Norm(x)), then x = x + MLP(Norm(x)).",
          "LayerNorm vs. RMSNorm.",
          "Position-wise Feed-Forward MLP: expanding C -> 4*C -> C."
        ],
        "code": "import torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(cfg.emb_dim)\n        self.attn = MultiHeadAttention(cfg.emb_dim, cfg.n_heads)\n        self.ln2 = nn.LayerNorm(cfg.emb_dim)\n        self.mlp = nn.Sequential(\n            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),\n            nn.GELU(),\n            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),\n        )\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        x = x + self.mlp(self.ln2(x))\n        return x",
        "quiz": {
          "question": "How do Residual Connections (x + Sublayer(x)) help train deep 96-layer Transformers?",
          "options": [
            "They create an unobstructed highway for gradients to flow backward without vanishing",
            "They reduce vocabulary size by half",
            "They eliminate the need for a tokenizer"
          ],
          "answerIndex": 0,
          "explanation": "Because d(x + f(x))/dx = 1 + df/dx, gradients always have a clean '+1' path straight back to early layers."
        },
        "pitfall": "Using BatchNorm instead of LayerNorm/RMSNorm!",
        "project": "Implement LayerNorm from scratch and verify it matches nn.LayerNorm to 1e-5.",
        "resourceTitle": "Sebastian Raschka — Chapter 4: Implementing a GPT Model from Scratch",
        "resourceNote": "Implements LayerNorm, GELU, shortcut connections, and the full TransformerBlock cleanly.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 4 TransformerBlock Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04"
          },
          {
            "type": "Community Guide",
            "label": "Pinecone Learning Center — Layer Normalization in Transformers Explained",
            "url": "https://www.pinecone.io/learn/batch-layer-normalization/"
          }
        ]
      },
      {
        "id": "p4-t6",
        "codeNum": "4.6",
        "animType": "transformer_block",
        "title": "Assembling the Full Decoder-Only GPT Architecture & Parameter Counting",
        "duration": "3 Days",
        "plainEnglish": "Now we snap all the Lego bricks together: token embeddings + positional embeddings -> N stacked TransformerBlocks -> final LayerNorm -> LM Head.",
        "liveExample": "GPT-2 Small (124M params), Gemma 2B, and Qwen 2.5 follow this exact end-to-end Decoder-Only blueprint.",
        "jargonTerms": [
          {
            "term": "nn.ModuleList",
            "def": "PyTorch's official list wrapper so stacked layers register properly for GPU & training."
          },
          {
            "term": "Weight Tying",
            "def": "Reusing the input embedding matrix as the output LM Head to save millions of params."
          }
        ],
        "why": "Connecting embeddings, N Transformer blocks, final LayerNorm, and the Language Model Head gives you a genuine GPT architecture.",
        "lessonSections": [
          {
            "heading": "1. Weight Tying Between Embedding and LM Head",
            "body": "Sharing self.tok_emb.weight = self.lm_head.weight saves vocab_size * emb_dim parameters (38.6M params in GPT-2!) and ties input/output semantic geometry.",
            "codeExample": "self.tok_emb.weight = self.lm_head.weight"
          }
        ],
        "concepts": [
          "Stacking N identical TransformerBlocks via nn.ModuleList.",
          "Final LayerNorm + Linear LM Head projecting (B, T, C) -> (B, T, vocab_size).",
          "Weight tying to save millions of parameters."
        ],
        "code": "import torch.nn as nn\n\nclass GPTModel(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)\n        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)\n        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])\n        self.ln_f = nn.LayerNorm(cfg.emb_dim)\n        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)\n        self.tok_emb.weight = self.lm_head.weight",
        "quiz": {
          "question": "What happens if you store Transformer blocks in a plain Python list [] instead of nn.ModuleList([])?",
          "options": [
            "PyTorch won't register their parameters, so model.parameters() and .to('cuda') will silently skip them",
            "Training runs 2x faster",
            "It automatically enables LoRA"
          ],
          "answerIndex": 0,
          "explanation": "nn.Module only tracks submodules stored in attributes or PyTorch containers like nn.ModuleList / nn.Sequential."
        },
        "pitfall": "Storing Transformer blocks in a plain Python list instead of nn.ModuleList!",
        "project": "Instantiate a ~10M-parameter mini-GPT and print exact parameter counts per component.",
        "resourceTitle": "Andrej Karpathy nanoGPT & freeCodeCamp Full LLM Build",
        "resourceNote": "Complete end-to-end GPTModel walkthrough with parameter breakdown.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — Official nanoGPT Repository (model.py)",
            "url": "https://github.com/karpathy/nanoGPT"
          },
          {
            "type": "Community Guide",
            "label": "Brendan Bycroft — LLM Visualization (Interactive 3D GPT Walkthrough)",
            "url": "https://bbycroft.net/llm"
          }
        ]
      }
    ]
  },
  {
    "id": "phase-5",
    "number": "05",
    "category": "training",
    "accent": "amber",
    "duration": "Week 9–10",
    "title": "Pretraining, Evaluation & Text Generation",
    "subtitle": "Train your custom GPT from random weights on real text, track loss & perplexity curves, and decode coherent completions.",
    "plainBigPicture": "A freshly built GPT starts with random weights—so its first output is pure gibberish. During 'Pretraining', we feed it real text, measure its surprise ('Perplexity'), warm up and decay its learning rate, and watch coherent English emerge!",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 5 Skills",
      "summary": "Every API slider and training stability trick in frontier AI labs comes directly from Phase 5:",
      "examples": [
        "The 'Temperature' and 'Top-P' parameters in OpenAI, Anthropic, and Ollama APIs execute the exact logit scaling & nucleus sorting code in Module 5.5.",
        "Gradient clipping (max_norm=1.0) and Cosine LR Warmup (Modules 5.2–5.3) keep multi-month training runs from diverging."
      ]
    },
    "jargonDecoder": [
      {
        "term": "Pretraining",
        "def": "Teaching a random model language from raw text"
      },
      {
        "term": "Perplexity",
        "def": "How confused the model is (1 = certain)"
      },
      {
        "term": "Overfitting",
        "def": "Memorizing training text instead of generalizing"
      },
      {
        "term": "Temperature",
        "def": "Creativity dial when picking the next word"
      },
      {
        "term": "Top-k / Top-p",
        "def": "Filtering out weird low-probability words"
      }
    ],
    "achievements": [
      "Pretrain your custom GPT from random weights on real text using gradient accumulation, gradient clipping, and cosine LR warmup.",
      "Measure model uncertainty via Perplexity and diagnose overfitting vs. underfitting from train/validation loss curves.",
      "Code an autoregressive text generator supporting Greedy, Temperature scaling, Top-k, and Top-p (Nucleus) sampling."
    ],
    "miniProject": "Pretrain a Mini-GPT on TinyShakespeare & Generate Text with Temperature/Top-k/Top-p",
    "topics": [
      {
        "id": "p5-t1",
        "codeNum": "5.1",
        "animType": "softmax_logits",
        "title": "Next-Token Prediction Objective & Perplexity Metric",
        "duration": "2 Days",
        "plainEnglish": "Taking e^loss gives 'Perplexity'—which means the model is roughly as uncertain as rolling an N-sided die to guess the next word. Lower perplexity = smarter language model!",
        "liveExample": "GPT-2 improved zero-shot WikiText perplexity from ~65 down to ~18 as parameter scale and data quality increased.",
        "jargonTerms": [
          {
            "term": "Perplexity",
            "def": "exp(loss)—effective number of equally likely next words the model considers."
          }
        ],
        "why": "Understanding how 3D logits (B, T, V) flatten into 2D cross-entropy inputs—and how Perplexity translates loss—is core to LLM evaluation.",
        "lessonSections": [
          {
            "heading": "1. Flattening (B, T, V) for Cross-Entropy",
            "body": "PyTorch's F.cross_entropy expects 2D predictions (N, Classes) and 1D targets (N,), so we flatten B and T together via .view(B * T, V).",
            "codeExample": "loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))\nppl = torch.exp(loss)"
          }
        ],
        "concepts": [
          "Flattening logits from (B, T, vocab_size) -> (B*T, vocab_size).",
          "Perplexity = exp(loss).",
          "Evaluating loss on a held-out validation split with torch.no_grad()."
        ],
        "code": "import torch, torch.nn.functional as F\nB, T, V = 2, 8, 50257\nlogits = torch.randn(B, T, V)\ntargets = torch.randint(0, V, (B, T))\nloss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))\nprint(f\"Loss: {loss.item():.3f} | Perplexity: {torch.exp(loss).item():.2f}\")",
        "quiz": {
          "question": "If a language model achieves a validation Cross-Entropy loss of 2.0, what is its Perplexity?",
          "options": [
            "e^2.0 ≈ 7.39",
            "2.0",
            "0.5"
          ],
          "answerIndex": 0,
          "explanation": "Perplexity is defined as exp(Cross-Entropy Loss), meaning the model is as uncertain as choosing uniformly among ~7.4 words."
        },
        "pitfall": "Computing validation loss with gradients enabled—always wrap evaluation loops in 'with torch.no_grad():'.",
        "project": "Write an evaluate_loss(model, val_loader) helper that reports average validation loss and perplexity.",
        "resourceTitle": "Hugging Face Transformers Docs — Perplexity of Fixed-Length Models",
        "resourceNote": "Explains the mathematical link between Cross-Entropy loss and Perplexity.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 5 Pretraining & Evaluation Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05"
          },
          {
            "type": "Blog / Docs",
            "label": "Hugging Face Docs — Perplexity of Fixed-Length Models",
            "url": "https://huggingface.co/docs/transformers/perplexity"
          }
        ]
      },
      {
        "id": "p5-t2",
        "codeNum": "5.2",
        "animType": "gradient_descent",
        "title": "Pretraining Loop on TinyShakespeare: Gradient Accumulation & Clipping",
        "duration": "4 Days",
        "plainEnglish": "'Gradient Accumulation' runs several small mini-batches and adds up their gradients before stepping! 'Gradient Clipping' caps giant gradient spikes so one weird sentence never blows up your weights.",
        "liveExample": "Llama 3 trained with global batch sizes of 4M–16M tokens using gradient accumulation across micro-batches.",
        "jargonTerms": [
          {
            "term": "Gradient Accumulation",
            "def": "Summing gradients across several mini-batches to simulate a big batch."
          },
          {
            "term": "Gradient Clipping",
            "def": "Capping giant gradient spikes (max_norm=1.0) to prevent training crashes."
          }
        ],
        "why": "Gradient accumulation lets your laptop/GPU simulate large batches without running out of VRAM.",
        "lessonSections": [
          {
            "heading": "1. Scaling Loss by grad_accum_steps",
            "body": "Because PyTorch sums gradients across .backward() calls, divide loss by grad_accum_steps so the accumulated gradient equals the true mean.",
            "codeExample": "(loss / grad_accum_steps).backward()"
          }
        ],
        "concepts": [
          "Gradient accumulation across micro-batches.",
          "Gradient norm clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).",
          "Checkpointing model + optimizer state dicts."
        ],
        "code": "import torch\n# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n# torch.save({\"model\": model.state_dict(), \"step\": 100}, \"ckpt.pt\")\nprint(\"Gradient clipping & checkpoint pattern ready\")",
        "quiz": {
          "question": "When accumulating gradients over 4 micro-batches before calling optimizer.step(), how should you scale each micro-batch loss?",
          "options": [
            "Divide loss by 4 before calling loss.backward()",
            "Multiply loss by 4",
            "Call optimizer.zero_grad() after every micro-batch"
          ],
          "answerIndex": 0,
          "explanation": "Dividing by grad_accum_steps ensures the summed gradients match the true average gradient of a 4x larger batch."
        },
        "pitfall": "Forgetting to divide loss by grad_accum_steps before calling loss.backward()!",
        "project": "Pretrain your mini-GPT on TinyShakespeare for 3,000 steps and watch validation loss drop below ~1.6.",
        "resourceTitle": "Andrej Karpathy — nanoGPT Training Loop & TinyShakespeare Dataset",
        "resourceNote": "Shows real-time loss progression and sample text evolving from gibberish to Shakespearean dialogue.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — nanoGPT train.py",
            "url": "https://github.com/karpathy/nanoGPT/blob/master/train.py"
          },
          {
            "type": "Community Guide",
            "label": "Hugging Face Ultra-Scale Playbook",
            "url": "https://huggingface.co/spaces/nanotron/ultrascale-playbook"
          }
        ]
      },
      {
        "id": "p5-t3",
        "codeNum": "5.3",
        "animType": "lr_cosine",
        "title": "Learning Rate Warmup & Cosine Decay Scheduling",
        "duration": "2 Days",
        "plainEnglish": "'Linear Warmup' gently ramps up the learning rate over the first few hundred steps, and 'Cosine Decay' smoothly slows down near the finish line so the model settles cleanly into a low-loss valley.",
        "liveExample": "Chinchilla, GPT-3, and Llama pretraining schedules all use linear warmup followed by cosine decay down to 10% of peak LR.",
        "jargonTerms": [
          {
            "term": "LR Warmup",
            "def": "Gradually increasing learning rate from 0 to peak over early steps."
          },
          {
            "term": "Cosine Decay",
            "def": "Smoothly lowering learning rate along a gentle wave toward the end of training."
          }
        ],
        "why": "Warming up the learning rate gradually avoids early AdamW variance spikes, while cosine decay settles into sharp minima.",
        "lessonSections": [
          {
            "heading": "1. Two-Stage Cosine Schedule",
            "body": "Linear ramp for step < warmup_steps, followed by 0.5 * (1 + cos(pi * progress)) decay down to min_lr.",
            "codeExample": "lr = get_lr(step, warmup_steps=100, max_steps=5000, max_lr=6e-4, min_lr=6e-5)"
          }
        ],
        "concepts": [
          "Linear warmup phase (first ~5% of steps).",
          "Cosine annealing decay down to min_lr.",
          "Updating param_group['lr'] dynamically each step."
        ],
        "code": "import math\n\ndef get_lr(step, warmup_steps, max_steps, max_lr, min_lr):\n    if step < warmup_steps:\n        return max_lr * (step + 1) / warmup_steps\n    if step >= max_steps:\n        return min_lr\n    progress = (step - warmup_steps) / (max_steps - warmup_steps)\n    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))\n\nprint(\"Step 0 LR:\", get_lr(0, 100, 1000, 6e-4, 6e-5))",
        "quiz": {
          "question": "Why do LLM training runs start with a low learning rate that warms up linearly?",
          "options": [
            "Because weights are random and AdamW's variance estimates are noisy at step 0, so a huge initial LR causes instability",
            "Because GPUs need time to heat up physically",
            "To skip the first epoch of data"
          ],
          "answerIndex": 0,
          "explanation": "Warmup stabilizes early attention and LayerNorm statistics before applying peak step sizes."
        },
        "pitfall": "Leaving peak learning rate constant for the entire run.",
        "project": "Plot your get_lr() schedule curve in matplotlib across 5,000 steps.",
        "resourceTitle": "Sebastian Raschka — Appendix D: Learning Rate Warmup & Cosine Decay",
        "resourceNote": "Covers learning rate warmup, cosine decay, and gradient clipping in detail.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Appendix D Training Bells & Whistles Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-D"
          },
          {
            "type": "Community Guide",
            "label": "MachineLearningMastery — Learning Rate Warmup & Cosine Annealing Guide",
            "url": "https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/"
          }
        ]
      },
      {
        "id": "p5-t4",
        "codeNum": "5.4",
        "animType": "gradient_descent",
        "title": "Diagnosing Overfitting, Underfitting & Loss Curves",
        "duration": "2 Days",
        "plainEnglish": "Plotting both Training Loss (practice exam) and Validation Loss (unseen final exam) reveals immediately whether your model is memorizing or genuinely learning patterns.",
        "liveExample": "ML teams track train/val loss divergence in MLflow or Weights & Biases to trigger early stopping before overfitting wastes GPU hours.",
        "jargonTerms": [
          {
            "term": "Overfitting",
            "def": "Train loss drops near 0 while validation loss rises (pure memorization)."
          },
          {
            "term": "Underfitting",
            "def": "Both train and validation loss stay high (model too small or undertrained)."
          }
        ],
        "why": "Reading train vs. validation loss curves tells you immediately whether your model needs more capacity, more data, or stronger regularization.",
        "lessonSections": [
          {
            "heading": "1. Reading the Gap Between Train and Validation Loss",
            "body": "When train loss continues falling while validation loss bends upward, your model has started memorizing training sequences verbatim.",
            "codeExample": "# Healthy: train_loss=1.62, val_loss=1.68\n# Overfit: train_loss=0.41, val_loss=2.45"
          }
        ],
        "concepts": [
          "Healthy convergence vs. Overfitting vs. Underfitting.",
          "Logging metrics cleanly to MLflow."
        ],
        "code": "# import mlflow\n# mlflow.log_metric(\"train_loss\", train_loss, step=step)\n# mlflow.log_metric(\"val_loss\", val_loss, step=step)\nprint(\"MLflow logging ready\")",
        "quiz": {
          "question": "Your training loss is 0.35 (very low), but your validation loss climbed from 1.70 to 2.60. What is happening?",
          "options": [
            "Overfitting — the model is memorizing training text instead of generalizing",
            "Underfitting — the model is too small",
            "Perfect convergence"
          ],
          "answerIndex": 0,
          "explanation": "Diverging train vs. validation loss is the classic signature of overfitting."
        },
        "pitfall": "Evaluating only training loss!",
        "project": "Log train and validation loss curves with matplotlib/MLflow on your TinyShakespeare run.",
        "resourceTitle": "Andrej Karpathy — A Recipe for Training Neural Networks",
        "resourceNote": "Timeless practical debugging checklist for diagnosing loss curves and silent bugs.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — A Recipe for Training Neural Networks",
            "url": "https://karpathy.github.io/2019/04/25/recipe/"
          },
          {
            "type": "Blog / Docs",
            "label": "MLflow Official Docs — Tracking PyTorch Experiments",
            "url": "https://mlflow.org/docs/latest/deep-learning/pytorch/index.html"
          }
        ]
      },
      {
        "id": "p5-t5",
        "codeNum": "5.5",
        "animType": "softmax_logits",
        "title": "Autoregressive Sampling Strategies: Greedy, Temperature, Top-k & Top-p (Nucleus)",
        "duration": "3 Days",
        "plainEnglish": "'Temperature' adjusts creativity by dividing logits before Softmax, while 'Top-k' and 'Top-p' block absurd tail words from ever being chosen.",
        "liveExample": "ChatGPT's API exposes exact 'temperature' and 'top_p' parameters backed by this exact Softmax + cumulative mask logic.",
        "jargonTerms": [
          {
            "term": "Temperature",
            "def": "Divides logits before Softmax: low (0.2) = focused; high (1.5) = creative."
          },
          {
            "term": "Top-k / Top-p",
            "def": "Keeps only the top k words or top p% probability mass before sampling."
          }
        ],
        "why": "Greedy decoding repeats loops, while unfiltered random sampling picks bizarre low-probability words. Temperature, Top-k, and Top-p strike the sweet spot.",
        "lessonSections": [
          {
            "heading": "1. How Temperature Reshapes Logits",
            "body": "Dividing logits by T < 1 exaggerates differences between the #1 word and runner-ups; dividing by T > 1 flattens probabilities closer to uniform.",
            "codeExample": "probs = F.softmax(logits / temperature, dim=-1)"
          }
        ],
        "concepts": [
          "Autoregressive loop cropping context to last max_len tokens.",
          "Temperature scaling (logits / T).",
          "Top-k and Top-p Nucleus sampling."
        ],
        "code": "import torch, torch.nn.functional as F\nlogits = torch.tensor([[5.2, 3.1, 2.3, 1.4, 0.1]])\ntemperature, top_p = 0.7, 0.9\nsorted_logits, sorted_idx = torch.sort(logits / temperature, descending=True)\ncum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)\nsorted_mask = cum_probs > top_p\nsorted_mask[..., 1:] = sorted_mask[..., :-1].clone()\nsorted_mask[..., 0] = False\nsorted_logits[sorted_mask] = float('-inf')\nprint(\"Filtered probs:\", F.softmax(sorted_logits, dim=-1))",
        "quiz": {
          "question": "What happens when you lower Temperature from 1.0 down to 0.2?",
          "options": [
            "The probability distribution sharpens around the highest-logit token, making outputs more deterministic",
            "All vocabulary words become equally likely",
            "The context window doubles"
          ],
          "answerIndex": 0,
          "explanation": "Dividing logits by a small number (< 1) amplifies the gap between the top logit and alternatives."
        },
        "pitfall": "Forgetting to slice context to the last max_len tokens (idx[:, -cfg.context_length:]) inside the generation loop!",
        "project": "Implement a unified generate(idx, max_new_tokens, temperature, top_k, top_p) method.",
        "resourceTitle": "Hugging Face Blog — How to Generate Text (Top-k & Top-p)",
        "resourceNote": "Visualizes probability bar charts before and after Temperature scaling and Top-k filtering.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Hugging Face Blog — How to Generate Text: Greedy, Beam, Top-k & Top-p",
            "url": "https://huggingface.co/blog/how-to-generate"
          },
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 5 Decoding Strategies Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05"
          }
        ]
      }
    ]
  },
  {
    "id": "phase-6",
    "number": "06",
    "category": "training",
    "accent": "teal",
    "duration": "Week 11–12",
    "title": "Fine-Tuning, Alignment & Local Deployment",
    "subtitle": "Load open pretrained weights into your own PyTorch code, fine-tune for chat instructions with LoRA, and deploy behind FastAPI + Ollama.",
    "plainBigPicture": "A base pretrained model is only an autocomplete engine—not yet a helpful Q&A assistant. In this final phase, you load open pretrained weights into your own code, teach the model to answer questions using 'SFT' and 'LoRA', and wrap it in a live web API.",
    "liveAiBridge": {
      "headline": "Where Live Running AIs Use Phase 6 Skills",
      "summary": "Every custom enterprise AI assistant and local coding copilot runs this exact post-training & serving stack:",
      "examples": [
        "Multi-tenant cloud platforms serve hundreds of custom fine-tuned models from a single base GPU by hot-swapping tiny 20MB LoRA adapter matrices (Module 6.3) per request.",
        "Ollama and llama.cpp serve 4-bit quantized GGUF checkpoints locally on consumer laptops (Modules 6.3 & 6.5)."
      ]
    },
    "jargonDecoder": [
      {
        "term": "SFT",
        "def": "Fine-tuning a base model on Q&A chat pairs"
      },
      {
        "term": "LoRA",
        "def": "Training tiny adapter matrices instead of all weights"
      },
      {
        "term": "Quantization",
        "def": "Compressing weights (32-bit -> 4-bit) to fit laptop RAM"
      },
      {
        "term": "DPO / RLHF",
        "def": "Aligning replies to human helpfulness preferences"
      },
      {
        "term": "FastAPI / Ollama",
        "def": "Serving your model locally as a REST API"
      }
    ],
    "achievements": [
      "Map open pretrained weights (such as GPT-2 124M) directly into your from-scratch PyTorch classes and verify exact logit parity.",
      "Fine-tune a base LLM into a Q&A assistant using Supervised Fine-Tuning (SFT) and memory-saving LoRA adapters.",
      "Deploy your custom model locally behind a FastAPI + Uvicorn REST endpoint and benchmark it alongside Ollama."
    ],
    "miniProject": "Fine-Tune a Small Model on Custom Q&A Data & Serve Locally via FastAPI + Uvicorn",
    "topics": [
      {
        "id": "p6-t1",
        "codeNum": "6.1",
        "animType": "transformer_block",
        "title": "Loading Open Pretrained Weights (GPT-2 / Llama) into Custom PyTorch Code",
        "duration": "3 Days",
        "plainEnglish": "Copy free open pretrained checkpoint numbers into your own from-scratch PyTorch classes—proving your architecture is 100% authentic.",
        "liveExample": "Inference engines like vLLM and llama.cpp map Hugging Face state_dict tensors into custom high-speed model classes using this exact key-matching process.",
        "jargonTerms": [
          {
            "term": "state_dict()",
            "def": "Python dictionary mapping every layer name to its saved weight tensor."
          }
        ],
        "why": "Pretraining from scratch on web-scale data costs millions, so real-world builders map pretrained checkpoint tensors directly into their own PyTorch architecture.",
        "lessonSections": [
          {
            "heading": "1. Verifying Tensor Shapes Before Copying",
            "body": "Always assert source_tensor.shape == target_tensor.shape when mapping external weights into your custom GPTModel.",
            "codeExample": "assert target.shape == source.shape, f'Mismatch: {target.shape} vs {source.shape}'"
          }
        ],
        "concepts": [
          "Inspecting state_dict() keys and tensor shapes.",
          "Handling Conv1D vs. nn.Linear weight transpositions.",
          "Verifying exact logit parity."
        ],
        "code": "# for k, v in list(model.state_dict().items())[:6]:\n#     print(f\"{k:35s} | {tuple(v.shape)}\")\nprint(\"Ready to map pretrained checkpoint tensors\")",
        "quiz": {
          "question": "Why do OpenAI's original GPT-2 linear weights need .T (transposition) when loaded into PyTorch nn.Linear?",
          "options": [
            "OpenAI stored Conv1D weights as (in_features, out_features), whereas PyTorch nn.Linear stores (out_features, in_features)",
            "To convert integers to strings",
            "To reduce parameter count by 50%"
          ],
          "answerIndex": 0,
          "explanation": "Checking .shape catches transposed weight layouts immediately before copying tensors."
        },
        "pitfall": "OpenAI's original GPT-2 checkpoints store linear weights transposed compared to PyTorch's nn.Linear!",
        "project": "Load pretrained GPT-2 Small (124M) weights into your from-scratch PyTorch GPTModel.",
        "resourceTitle": "Sebastian Raschka — Loading OpenAI Pretrained Weights & nanoGPT",
        "resourceNote": "Step-by-step weight mapping function from pretrained tensors into custom GPTModel layers.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 5.5 Loading GPT-2 Weights Notebook",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05"
          },
          {
            "type": "Community Guide",
            "label": "Andrej Karpathy — nanoGPT from_pretrained GPT-2 Implementation",
            "url": "https://github.com/karpathy/nanoGPT/blob/master/model.py"
          }
        ]
      },
      {
        "id": "p6-t2",
        "codeNum": "6.2",
        "animType": "sliding_window",
        "title": "Supervised Fine-Tuning (SFT) & Instruction Dataset Formatting",
        "duration": "3 Days",
        "plainEnglish": "'Supervised Fine-Tuning (SFT)' trains a base autocomplete model on curated Instruction -> Answer pairs so it behaves like a helpful assistant.",
        "liveExample": "Meta's Llama-3-Instruct and OpenAI's GPT-3.5-Turbo were transformed from document completers into chat assistants via SFT.",
        "jargonTerms": [
          {
            "term": "SFT",
            "def": "Supervised Fine-Tuning on formatted (Instruction, Response) examples."
          },
          {
            "term": "ignore_index=-100",
            "def": "Tells PyTorch's loss function to skip padding tokens when grading."
          }
        ],
        "why": "SFT teaches a base model to follow instructions and answer as an assistant.",
        "lessonSections": [
          {
            "heading": "1. Masking Padding & Prompt Tokens with -100",
            "body": "Setting target IDs to -100 on padding tokens instructs F.cross_entropy to ignore them so loss focuses purely on assistant responses.",
            "codeExample": "loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)"
          }
        ],
        "concepts": [
          "Structuring instruction-input-response JSON records into prompt templates.",
          "Target padding with ignore_index = -100.",
          "Using a low learning rate (2e-5) for 2–3 epochs."
        ],
        "code": "import torch.nn.functional as F\n# loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)\nprint(\"SFT loss masking configured with ignore_index=-100\")",
        "quiz": {
          "question": "Why do SFT collators fill padding positions in the target tensor with -100?",
          "options": [
            "PyTorch's F.cross_entropy has ignore_index=-100 by default, skipping padded slots in loss computation",
            "Because -100 is the token ID for space",
            "To double the learning rate on padding"
          ],
          "answerIndex": 0,
          "explanation": "Ignoring padding tokens prevents the model from wasting capacity predicting meaningless pad tokens."
        },
        "pitfall": "Training SFT for 20+ epochs on a tiny dataset causes catastrophic forgetting.",
        "project": "Curate a 100-item IT/Python Q&A dataset and fine-tune your model to answer domain questions.",
        "resourceTitle": "Sebastian Raschka — Chapter 7: Instruction Fine-Tuning & Hugging Face SFT",
        "resourceNote": "Covers custom collate functions, -100 padding masks, and instruction tuning end to end.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Chapter 7 Instruction Fine-Tuning Code",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07"
          },
          {
            "type": "Blog / Docs",
            "label": "Hugging Face LLM Course — Chapter 3: Fine-Tuning a Pretrained Model",
            "url": "https://huggingface.co/learn/llm-course/chapter3/1"
          }
        ]
      },
      {
        "id": "p6-t3",
        "codeNum": "6.3",
        "animType": "lora_lowrank",
        "title": "Parameter-Efficient Fine-Tuning (LoRA) & Quantization Basics",
        "duration": "3 Days",
        "plainEnglish": "'LoRA' freezes the original brain and attaches tiny side-matrices (A and B) that learn only the new task—cutting memory by 90%! 'Quantization' compresses weights to 4-bit integers.",
        "liveExample": "Apple Intelligence on iPhone/Mac runs a ~3B parameter on-device LLM using 2-bit/4-bit quantization and task-specific LoRA adapters.",
        "jargonTerms": [
          {
            "term": "LoRA",
            "def": "Low-Rank Adaptation—training tiny side matrices (A @ B) while base weights stay frozen."
          },
          {
            "term": "Quantization",
            "def": "Compressing model weights from 32-bit floats to 8-bit or 4-bit numbers."
          }
        ],
        "why": "LoRA freezes base weights and trains tiny low-rank adapter matrices (~1% of parameters), while quantization shrinks RAM 4x.",
        "lessonSections": [
          {
            "heading": "1. Zero-Initializing LoRA Matrix B",
            "body": "Initializing B = 0 ensures ΔW = A @ B = 0 at step 0, so fine-tuning begins from the exact pretrained model without random noise.",
            "codeExample": "self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)\nself.B = nn.Parameter(torch.zeros(rank, out_dim))"
          }
        ],
        "concepts": [
          "LoRA math: freezing W and learning ΔW = A @ B with rank r = 8 or 16.",
          "Initializing B = 0 and A ~ N(0, sigma^2).",
          "Quantization: FP32 -> BF16 -> INT8 -> INT4/GGUF."
        ],
        "code": "import torch, torch.nn as nn\n\nclass LoRALayer(nn.Module):\n    def __init__(self, in_dim, out_dim, rank=8, alpha=16):\n        super().__init__()\n        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)\n        self.B = nn.Parameter(torch.zeros(rank, out_dim))\n        self.scale = alpha / rank\n    def forward(self, x):\n        return (x @ self.A @ self.B) * self.scale",
        "quiz": {
          "question": "Why is LoRA matrix B initialized to all zeros while matrix A is initialized with small random values?",
          "options": [
            "So ΔW = A @ B equals 0 at step 0, preserving the exact pretrained model behavior at the start of training",
            "Because zero matrices multiply faster",
            "To freeze matrix A"
          ],
          "answerIndex": 0,
          "explanation": "If both A and B were random, step 0 would inject random noise into every adapted layer."
        },
        "pitfall": "Forgetting to freeze base model parameters (param.requires_grad = False) before training LoRA!",
        "project": "Implement a LinearWithLoRA wrapper in pure PyTorch and verify trainable parameter count drops by >95%.",
        "resourceTitle": "Sebastian Raschka — LoRA From Scratch in PyTorch (Appendix E)",
        "resourceNote": "Implements LoRA from scratch in pure PyTorch with clear rank/alpha explanations.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — Appendix E: LoRA From Scratch in PyTorch",
            "url": "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E"
          },
          {
            "type": "Community Guide",
            "label": "Maarten Grootendorst — A Visual Guide to Quantization",
            "url": "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization"
          }
        ]
      },
      {
        "id": "p6-t4",
        "codeNum": "6.4",
        "animType": "softmax_logits",
        "title": "Evaluation Benchmarks, Hallucination Checks & Alignment (DPO Intuition)",
        "duration": "2 Days",
        "plainEnglish": "Before shipping a model, we run automated 'pytest' checks on unseen prompts with temperature=0, and use preference alignment ('DPO') to reward honest, helpful replies.",
        "liveExample": "Zephyr, Llama-3-Instruct, and Qwen-Chat all apply Direct Preference Optimization (DPO) after SFT to reduce hallucinations and improve helpfulness.",
        "jargonTerms": [
          {
            "term": "Hallucination",
            "def": "When an LLM confidently generates false facts not supported by context."
          },
          {
            "term": "DPO / RLHF",
            "def": "Training the model to prefer human-approved answers over rejected ones."
          }
        ],
        "why": "Evaluating quality requires automated benchmarks, deterministic rubric checks, and preference alignment (RLHF / DPO).",
        "lessonSections": [
          {
            "heading": "1. Deterministic Evaluation with temperature=0.0",
            "body": "Always set model.eval() and greedy decoding (temperature=0.0) when running automated pytest regressions so test results are 100% reproducible.",
            "codeExample": "@torch.no_grad()\ndef evaluate_prompt(model, ids):\n    model.eval()"
          }
        ],
        "concepts": [
          "Automated evaluation: held-out task accuracy and schema validation.",
          "Direct Preference Optimization (DPO) on (chosen, rejected) pairs.",
          "Deterministic inference (temperature=0.0)."
        ],
        "code": "import torch\n@torch.no_grad()\ndef evaluate_prompt(model, prompt_ids):\n    model.eval()\n    return prompt_ids",
        "quiz": {
          "question": "What training data format does Direct Preference Optimization (DPO) require?",
          "options": [
            "Triplets of (Prompt, Chosen Good Response, Rejected Bad Response)",
            "Unlabeled raw text only",
            "Image-to-audio pairs"
          ],
          "answerIndex": 0,
          "explanation": "DPO directly increases the log-probability margin between preferred (chosen) and dispreferred (rejected) completions."
        },
        "pitfall": "Evaluating your model on prompts that accidentally leaked into your SFT training split!",
        "project": "Build a 20-prompt automated evaluation script using pytest.",
        "resourceTitle": "Sebastian Raschka — DPO Alignment & LLM Evaluation",
        "resourceNote": "Demonstrates automated scoring of fine-tuned instruction responses.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Sebastian Raschka — LLM Alignment: RLHF vs. DPO Explained",
            "url": "https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives"
          },
          {
            "type": "Blog / Docs",
            "label": "Hugging Face TRL Docs — Direct Preference Optimization (DPO)",
            "url": "https://huggingface.co/docs/trl/main/en/dpo_trainer"
          }
        ]
      },
      {
        "id": "p6-t5",
        "codeNum": "6.5",
        "animType": "serving_api",
        "title": "Serving Your Model Locally with FastAPI, Uvicorn & Ollama",
        "duration": "3 Days",
        "plainEnglish": "'FastAPI' creates a clean HTTP endpoint (/v1/generate) where any app can send a prompt and get JSON back, 'Uvicorn' runs that web server locally, and 'Ollama' runs local GGUF models.",
        "liveExample": "Ollama exposes an OpenAI-compatible localhost REST API on port 11434 using this exact JSON request/response contract.",
        "jargonTerms": [
          {
            "term": "FastAPI",
            "def": "Modern Python framework for building fast, auto-documented REST APIs."
          },
          {
            "term": "Uvicorn",
            "def": "Lightweight web server that runs your FastAPI app on localhost."
          }
        ],
        "why": "Completing the engineering loop: package your PyTorch model behind a typed REST API using FastAPI + Uvicorn.",
        "lessonSections": [
          {
            "heading": "1. Load Weights Once at Startup",
            "body": "Instantiate your GPTModel and load its checkpoint once at module startup so every POST /v1/generate request runs in milliseconds.",
            "codeExample": "@app.post('/v1/generate')\ndef generate_endpoint(req: GenerateRequest):\n    with torch.no_grad():\n        return {'output': 'generated text'}"
          }
        ],
        "concepts": [
          "Loading model weights once at FastAPI startup.",
          "Validating request/response JSON payloads with Pydantic BaseModel.",
          "Running local quantized models via Ollama."
        ],
        "code": "from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI(title=\"My From-Scratch LLM API\")\n\nclass GenerateRequest(BaseModel):\n    prompt: str\n    max_tokens: int = 60\n    temperature: float = 0.7",
        "quiz": {
          "question": "Where should you load your PyTorch model checkpoint inside a FastAPI application?",
          "options": [
            "Once at application startup outside the route handler, so requests don't reload gigabytes from disk",
            "Inside the @app.post('/generate') function on every request",
            "Inside the Pydantic BaseModel class"
          ],
          "answerIndex": 0,
          "explanation": "Loading weights once into GPU/CPU memory at startup keeps per-request inference latency down to milliseconds."
        },
        "pitfall": "Reloading the PyTorch checkpoint file inside the /generate route handler on every request!",
        "project": "Launch your custom LLM server with 'uvicorn app:app --reload' and test it via Swagger UI (/docs).",
        "resourceTitle": "FastAPI Official Docs & Ollama Quickstart",
        "resourceNote": "Brings together PyTorch, FastAPI, Uvicorn, requests, and Ollama from your environment.",
        "links": [
          {
            "type": "Community Guide",
            "label": "Real Python — Using FastAPI to Build Python Web APIs",
            "url": "https://realpython.com/fastapi-python-web-apis/"
          },
          {
            "type": "Blog / Docs",
            "label": "FastAPI Official Tutorial — First Steps & Automatic /docs UI",
            "url": "https://fastapi.tiangolo.com/tutorial/first-steps/"
          },
          {
            "type": "Blog / Docs",
            "label": "Ollama Official Quickstart & REST API Documentation",
            "url": "https://docs.ollama.com/quickstart"
          }
        ]
      }
    ]
  }
];

const ALL_TOPICS = ROADMAP_PHASES.flatMap((phase) => phase.topics.map((t) => ({ ...t, phaseId: phase.id, phaseNumber: phase.number, phaseTitle: phase.title })));

const STORAGE_KEY = "llm_scratch_roadmap_completed_v2";

function getAccentClasses(accent) {
    const map = {
      teal: {
        badgeBg: "bg-teal-50 text-teal-800 border-teal-200",
        nodeRing: "border-teal-500 text-teal-700 bg-white",
        bar: "bg-teal-500",
        pillHover: "hover:border-teal-300"
      },
      indigo: {
        badgeBg: "bg-indigo-50 text-indigo-800 border-indigo-200",
        nodeRing: "border-indigo-500 text-indigo-700 bg-white",
        bar: "bg-indigo-500",
        pillHover: "hover:border-indigo-300"
      },
      amber: {
        badgeBg: "bg-amber-50 text-amber-800 border-amber-200",
        nodeRing: "border-amber-500 text-amber-700 bg-white",
        bar: "bg-amber-500",
        pillHover: "hover:border-amber-300"
      }
    };
    return map[accent] || map.teal;
  }

function MotionBPELab() {
    const [mode, setMode] = (0, import_react.useState)("stepper");
    const [stepIdx, setStepIdx] = (0, import_react.useState)(0);
    const [customText, setCustomText] = (0, import_react.useState)("unbelievable tokenization");
    const current = BPE_STEP_SCENARIOS[stepIdx];
    const getTokenStyle = (type) => {
      switch (type) {
        case "raw":
          return "bg-slate-100 border-slate-300 text-slate-800";
        case "char":
          return "bg-sky-50 border-sky-200 text-sky-800";
        case "merged":
          return "bg-emerald-50 border-emerald-300 text-emerald-800 ring-2 ring-emerald-500/20";
        case "final-prefix":
          return "bg-indigo-50 border-indigo-300 text-indigo-800 shadow-xs";
        case "final-stem":
          return "bg-amber-50 border-amber-300 text-amber-800 shadow-xs";
        case "final-suffix":
          return "bg-emerald-50 border-emerald-300 text-emerald-800 shadow-xs";
        default:
          return "bg-white border-slate-200 text-slate-800";
      }
    };
    const liveCustomTokens = (0, import_react.useMemo)(() => {
      const clean = (customText || "hello").slice(0, 48);
      const rules = [
        ["u", "n"],
        ["l", "e"],
        ["a", "b"],
        ["ab", "le"],
        ["t", "i"],
        ["ti", "on"],
        ["o", "k"],
        ["t", "ok"],
        ["e", "n"],
        ["i", "z"],
        ["at", "ion"]
      ];
      let syms = clean.split("");
      rules.forEach(([a, b]) => {
        const next = [];
        let i = 0;
        while (i < syms.length) {
          if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
            next.push(a + b);
            i += 2;
          } else {
            next.push(syms[i]);
            i += 1;
          }
        }
        syms = next;
      });
      return syms.map((tok, i) => {
        let id = tok.length === 1 ? tok.charCodeAt(0) : 256;
        if (tok.length > 1) {
          for (let c = 0; c < tok.length; c++)
            id = (id * 31 + tok.charCodeAt(c)) % 49e3 + 256;
        }
        return { id: `c-${i}-${tok}`, text: tok, tokenId: id, bytes: new TextEncoder().encode(tok).length };
      });
    }, [customText]);
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md" }, "Interactive Lab \u2022 motion/react"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-extrabold text-[#17394D]" }, "Byte-Pair Encoding (BPE) Tokenization Stepper")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Step through how 12 raw UTF-8 characters merge into 3 dense subword IDs\u2014or test your own custom sentence.")), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "inline-flex rounded-lg bg-slate-200/70 p-0.5 text-xs font-semibold" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setMode("stepper"),
        className: `px-2.5 py-1 rounded-md transition ${mode === "stepper" ? "bg-white text-[#17394D] shadow-2xs" : "text-slate-600"}`
      },
      "5-Step Merge Walkthrough"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setMode("sandbox"),
        className: `px-2.5 py-1 rounded-md transition ${mode === "sandbox" ? "bg-white text-[#17394D] shadow-2xs" : "text-slate-600"}`
      },
      "Live Text Sandbox"
    )))), mode === "stepper" ? /* @__PURE__ */ import_react.default.createElement(import_react.default.Fragment, null, /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-5 gap-1.5 px-6 pt-4" }, BPE_STEP_SCENARIOS.map((s, i) => /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        key: s.step,
        type: "button",
        onClick: () => setStepIdx(i),
        className: `h-1.5 rounded-full transition-all duration-300 ${i <= stepIdx ? "bg-indigo-600" : "bg-slate-200 hover:bg-slate-300"}`,
        title: `Step ${i}: ${s.title}`
      }
    ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-6 space-y-5" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-3" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold text-indigo-600" }, "Step ", current.step, " of ", BPE_STEP_SCENARIOS.length - 1, " \u2014 ", current.title), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs sm:text-sm text-slate-600 mt-0.5" }, current.desc)), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setStepIdx(0),
        disabled: stepIdx === 0,
        className: "px-2.5 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-100 disabled:opacity-40"
      },
      "Reset"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setStepIdx((s) => Math.max(0, s - 1)),
        disabled: stepIdx === 0,
        className: "px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40"
      },
      "\u2190 Prev"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setStepIdx((s) => Math.min(BPE_STEP_SCENARIOS.length - 1, s + 1)),
        disabled: stepIdx === BPE_STEP_SCENARIOS.length - 1,
        className: "px-3.5 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 shadow-2xs"
      },
      "Next Merge \u2192"
    ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "min-h-[116px] p-5 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-wrap items-center justify-center gap-2.5" }, current.tokens.map((tok) => /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        key: tok.id,
        className: `px-3.5 py-2 rounded-xl border-2 font-mono flex flex-col items-center transition-all duration-300 hover:-translate-y-0.5 ${getTokenStyle(
          tok.type
        )}`
      },
      /* @__PURE__ */ import_react.default.createElement("span", { className: "text-base font-bold tracking-tight" }, '"', tok.text, '"'),
      /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-1.5 mt-1 text-[10px] opacity-80" }, /* @__PURE__ */ import_react.default.createElement("span", null, "ID: ", tok.tokenId), /* @__PURE__ */ import_react.default.createElement("span", null, "\u2022"), /* @__PURE__ */ import_react.default.createElement("span", null, tok.bytes, "B"))
    ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 text-xs" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2 text-slate-600" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-semibold" }, "Active Rule:"), /* @__PURE__ */ import_react.default.createElement("code", { className: "font-mono font-semibold text-slate-800 bg-slate-100 px-2 py-0.5 rounded" }, current.mergeRule)), /* @__PURE__ */ import_react.default.createElement("div", { className: "text-slate-500 font-mono" }, "Tensor Input:", " ", /* @__PURE__ */ import_react.default.createElement("span", { className: "text-indigo-600 font-bold" }, "[", current.tokens.map((t) => t.tokenId).join(", "), "]"))))) : /* @__PURE__ */ import_react.default.createElement("div", { className: "p-6 space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col sm:flex-row gap-2" }, /* @__PURE__ */ import_react.default.createElement(
      "input",
      {
        type: "text",
        value: customText,
        onChange: (e) => setCustomText(e.target.value),
        placeholder: "Type any phrase to tokenize...",
        className: "flex-1 px-3.5 py-2 rounded-xl border border-slate-300 text-xs sm:text-sm font-mono text-[#17394D]"
      }
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 font-mono text-xs text-slate-600 flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Compression:"), /* @__PURE__ */ import_react.default.createElement("strong", { className: "text-indigo-700" }, customText.length, " chars \u2192 ", liveCustomTokens.length, " tokens"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "min-h-[100px] p-4 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-wrap items-center justify-center gap-2" }, liveCustomTokens.map((tok) => /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        key: tok.id,
        className: `px-3 py-1.5 rounded-xl border-2 font-mono flex flex-col items-center ${tok.text.length > 1 ? "bg-emerald-50 border-emerald-300 text-emerald-900" : "bg-sky-50 border-sky-200 text-sky-900"}`
      },
      /* @__PURE__ */ import_react.default.createElement("span", { className: "text-sm font-bold" }, '"', tok.text === " " ? "\u2423" : tok.text, '"'),
      /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] opacity-80" }, "ID:", tok.tokenId, " \u2022 ", tok.bytes, "B")
    )))));
  }

function ThreeDEmbeddingLab() {
    const [rotY, setRotY] = (0, import_react.useState)(0.55);
    const [rotX, setRotX] = (0, import_react.useState)(0.35);
    const [isDragging, setIsDragging] = (0, import_react.useState)(false);
    const [hoveredNode, setHoveredNode] = (0, import_react.useState)(null);
    const [autoSpin, setAutoSpin] = (0, import_react.useState)(true);
    const [showSinusoid, setShowSinusoid] = (0, import_react.useState)(false);
    const dragStart = (0, import_react.useRef)({ x: 0, y: 0 });
    (0, import_react.useEffect)(() => {
      if (!autoSpin || isDragging)
        return;
      const interval = setInterval(() => setRotY((r) => r + 8e-3), 30);
      return () => clearInterval(interval);
    }, [autoSpin, isDragging]);
    const projectPoint = (x, y, z) => {
      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const x1 = x * cosY - z * sinY;
      const z1 = z * cosY + x * sinY;
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = z1 * cosX + y * sinX;
      const fov = 4.6;
      const scale = fov / (fov + z2 + 2.2);
      return {
        px: 260 + x1 * 100 * scale,
        py: 155 - y2 * 100 * scale,
        scale,
        depth: z2
      };
    };
    const projectedNodes = (0, import_react.useMemo)(() => {
      return EMBEDDING_3D_NODES.map((node, idx) => {
        const pxOffset = showSinusoid ? Math.cos(idx * 1.1) * 0.22 : 0;
        const pyOffset = showSinusoid ? Math.sin(idx * 1.1) * 0.22 : 0;
        return __spreadProps(__spreadValues({}, node), {
          proj: projectPoint(node.x + pxOffset, node.y + pyOffset, node.z)
        });
      }).sort((a, b) => b.proj.depth - a.proj.depth);
    }, [rotX, rotY, showSinusoid]);
    const nodeMap = (0, import_react.useMemo)(() => {
      const map = {};
      projectedNodes.forEach((n) => map[n.id] = n);
      return map;
    }, [projectedNodes]);
    const origin = projectPoint(0, 0, 0);
    const xAxisPos = projectPoint(2.2, 0, 0);
    const xAxisNeg = projectPoint(-2.2, 0, 0);
    const yAxisPos = projectPoint(0, 2, 0);
    const yAxisNeg = projectPoint(0, -2, 0);
    const zAxisPos = projectPoint(0, 0, 2.2);
    const zAxisNeg = projectPoint(0, 0, -2.2);
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-700 rounded-md" }, "Interactive Lab \u2022 @react-three/fiber"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-extrabold text-[#17394D]" }, "3D Semantic Embedding Space (PCA Projection)")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Drag to orbit 3D vector space, hover nodes to inspect coordinates, and toggle Positional Encoding shifts.")), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setShowSinusoid((s) => !s),
        className: `px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${showSinusoid ? "bg-teal-50 border-teal-300 text-teal-800" : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"}`
      },
      "Positional Offset: ",
      showSinusoid ? "ON" : "OFF"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setAutoSpin((s) => !s),
        className: `px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${autoSpin ? "bg-indigo-50 border-indigo-200 text-indigo-700" : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"}`
      },
      autoSpin ? "Auto-Orbit: ON" : "Auto-Orbit: OFF"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          setRotX(0.35);
          setRotY(0.55);
        },
        className: "px-2.5 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50"
      },
      "Reset"
    ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-5 space-y-3" }, /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        onPointerDown: (e) => {
          setIsDragging(true);
          setAutoSpin(false);
          dragStart.current = { x: e.clientX, y: e.clientY };
        },
        onPointerMove: (e) => {
          if (!isDragging)
            return;
          const dx = e.clientX - dragStart.current.x;
          const dy = e.clientY - dragStart.current.y;
          dragStart.current = { x: e.clientX, y: e.clientY };
          setRotY((r) => r + dx * 0.01);
          setRotX((r) => Math.max(-1.2, Math.min(1.2, r + dy * 0.01)));
        },
        onPointerUp: () => setIsDragging(false),
        onPointerLeave: () => setIsDragging(false),
        className: "relative w-full h-[310px] rounded-xl bg-slate-50 border border-slate-200 overflow-hidden cursor-grab active:cursor-grabbing select-none"
      },
      /* @__PURE__ */ import_react.default.createElement("svg", { viewBox: "0 0 520 310", className: "w-full h-full" }, /* @__PURE__ */ import_react.default.createElement("defs", null, /* @__PURE__ */ import_react.default.createElement("radialGradient", { id: "embedBgGlow", cx: "50%", cy: "50%", r: "50%" }, /* @__PURE__ */ import_react.default.createElement("stop", { offset: "0%", stopColor: "#ffffff" }), /* @__PURE__ */ import_react.default.createElement("stop", { offset: "100%", stopColor: "#f1f5f9" }))), /* @__PURE__ */ import_react.default.createElement("rect", { width: "520", height: "310", fill: "url(#embedBgGlow)" }), /* @__PURE__ */ import_react.default.createElement("line", { x1: xAxisNeg.px, y1: xAxisNeg.py, x2: xAxisPos.px, y2: xAxisPos.py, stroke: "#cbd5e1", strokeWidth: "1.2" }), /* @__PURE__ */ import_react.default.createElement("line", { x1: yAxisNeg.px, y1: yAxisNeg.py, x2: yAxisPos.px, y2: yAxisPos.py, stroke: "#cbd5e1", strokeWidth: "1.2" }), /* @__PURE__ */ import_react.default.createElement("line", { x1: zAxisNeg.px, y1: zAxisNeg.py, x2: zAxisPos.px, y2: zAxisPos.py, stroke: "#cbd5e1", strokeWidth: "1.2" }), /* @__PURE__ */ import_react.default.createElement("text", { x: xAxisPos.px + 6, y: xAxisPos.py + 4, fontSize: "10", fill: "#64748b", fontFamily: "monospace" }, "Dim-1 (Domain)"), /* @__PURE__ */ import_react.default.createElement("text", { x: yAxisPos.px - 28, y: yAxisPos.py - 8, fontSize: "10", fill: "#64748b", fontFamily: "monospace" }, "Dim-2 (Abstraction)"), /* @__PURE__ */ import_react.default.createElement("text", { x: zAxisPos.px + 6, y: zAxisPos.py + 4, fontSize: "10", fill: "#64748b", fontFamily: "monospace" }, "Dim-3"), EMBEDDING_SIMILARITY_LINKS.map((link) => {
        const a = nodeMap[link.from];
        const b = nodeMap[link.to];
        if (!a || !b)
          return null;
        const midX = (a.proj.px + b.proj.px) / 2;
        const midY = (a.proj.py + b.proj.py) / 2;
        return /* @__PURE__ */ import_react.default.createElement("g", { key: `${link.from}-${link.to}` }, /* @__PURE__ */ import_react.default.createElement(
          "line",
          {
            x1: a.proj.px,
            y1: a.proj.py,
            x2: b.proj.px,
            y2: b.proj.py,
            stroke: link.color,
            strokeWidth: "2",
            strokeDasharray: "4 4",
            opacity: "0.75"
          }
        ), /* @__PURE__ */ import_react.default.createElement("rect", { x: midX - 24, y: midY - 9, width: "48", height: "16", rx: "4", fill: "#ffffff", stroke: link.color, strokeWidth: "1" }), /* @__PURE__ */ import_react.default.createElement("text", { x: midX, y: midY + 2, textAnchor: "middle", fontSize: "9", fontWeight: "bold", fill: link.color, fontFamily: "monospace" }, "cos:", link.score));
      }), projectedNodes.map((node) => {
        const isHovered = (hoveredNode == null ? void 0 : hoveredNode.id) === node.id;
        const radius = (isHovered ? 10 : 7.5) * node.proj.scale;
        return /* @__PURE__ */ import_react.default.createElement(
          "g",
          {
            key: node.id,
            onMouseEnter: () => setHoveredNode(node),
            onMouseLeave: () => setHoveredNode(null),
            className: "cursor-pointer"
          },
          /* @__PURE__ */ import_react.default.createElement(
            "line",
            {
              x1: origin.px,
              y1: origin.py,
              x2: node.proj.px,
              y2: node.proj.py,
              stroke: node.color,
              strokeWidth: isHovered ? "1.5" : "0.75",
              opacity: isHovered ? "0.45" : "0.18"
            }
          ),
          /* @__PURE__ */ import_react.default.createElement("circle", { cx: node.proj.px, cy: node.proj.py, r: radius + 5, fill: node.color, opacity: isHovered ? 0.25 : 0.12 }),
          /* @__PURE__ */ import_react.default.createElement("circle", { cx: node.proj.px, cy: node.proj.py, r: radius, fill: node.color, stroke: "#ffffff", strokeWidth: "2" }),
          /* @__PURE__ */ import_react.default.createElement(
            "text",
            {
              x: node.proj.px,
              y: node.proj.py - radius - 6,
              textAnchor: "middle",
              fontSize: isHovered ? "13" : "11.5",
              fontWeight: "700",
              fill: "#0f172a"
            },
            node.word
          )
        );
      })),
      /* @__PURE__ */ import_react.default.createElement("div", { className: "absolute top-3 left-3 px-2.5 py-1 rounded-md bg-white/90 backdrop-blur-xs border border-slate-200 text-[11px] text-slate-600 font-medium shadow-2xs" }, "Drag canvas to rotate 3D vector space \u2022 Hover nodes for coordinates"),
      hoveredNode && /* @__PURE__ */ import_react.default.createElement("div", { className: "absolute bottom-3 right-3 p-3 rounded-xl bg-white/95 backdrop-blur-xs border border-slate-200 shadow-md text-xs space-y-1 min-w-[205px]" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-slate-900 text-sm" }, '"', hoveredNode.word, '"'), /* @__PURE__ */ import_react.default.createElement(
        "span",
        {
          className: "px-2 py-0.5 rounded text-[10px] font-semibold text-white",
          style: { backgroundColor: hoveredNode.color }
        },
        hoveredNode.cluster
      )), /* @__PURE__ */ import_react.default.createElement("div", { className: "text-slate-500 font-mono text-[11px]" }, "3D PCA: ", /* @__PURE__ */ import_react.default.createElement("span", { className: "text-slate-800 font-semibold" }, hoveredNode.vec)), /* @__PURE__ */ import_react.default.createElement("div", { className: "text-slate-400 text-[10px]" }, "Full Embedding: R^768 float32"))
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 pt-1" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-4" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-full bg-indigo-600" }), " Royalty Cluster"), /* @__PURE__ */ import_react.default.createElement("span", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-full bg-emerald-600" }), " Deep Learning Cluster"), /* @__PURE__ */ import_react.default.createElement("span", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-full bg-amber-500" }), " Architecture Cluster")), /* @__PURE__ */ import_react.default.createElement("span", { className: "font-mono text-[11px] text-slate-500" }, "cos(\u03B8) = (A\xB7B) / (||A|| ||B||)"))));
  }

function VisxAttentionHeatmapLab() {
    const [causalMask, setCausalMask] = (0, import_react.useState)(true);
    const [activeRow, setActiveRow] = (0, import_react.useState)(2);
    const [hoveredCell, setHoveredCell] = (0, import_react.useState)({ r: 2, c: 1 });
    const matrix = (0, import_react.useMemo)(() => {
      return RAW_ATTENTION_SCORES.map((row, rIdx) => {
        if (!causalMask)
          return row;
        const unmaskedSum = row.slice(0, rIdx + 1).reduce((acc, v) => acc + v, 0);
        return row.map((val, cIdx) => {
          if (cIdx > rIdx)
            return null;
          return Number((val / unmaskedSum).toFixed(2));
        });
      });
    }, [causalMask]);
    const activeWeights = matrix[activeRow];
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md" }, "Interactive Lab \u2022 @visx/heatmap"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-extrabold text-[#17394D]" }, "Causal Self-Attention Matrix")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Formula: ", /* @__PURE__ */ import_react.default.createElement("code", { className: "font-mono text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded" }, "Softmax(QK\u1D40 / \u221Ad\u2096 + M)"))), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setCausalMask((m) => !m),
        className: `px-3.5 py-1.5 rounded-lg text-xs font-bold border transition shadow-2xs ${causalMask ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"}`
      },
      "Apply Causal Mask (Decoder): ",
      causalMask ? "ON" : "OFF"
    )), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "lg:col-span-7" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex justify-between px-2" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Rows: Query (Q)"), /* @__PURE__ */ import_react.default.createElement("span", null, "Columns: Key (K)")), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-5 gap-2 font-mono text-xs bg-slate-50 p-4 rounded-xl border border-slate-200/80" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-center text-[10px] text-slate-400 font-sans" }, "Q \\ K"), ATTN_TOKENS.map((tok, cIdx) => /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        key: `col-${tok}`,
        className: `text-center font-bold py-1.5 rounded-md transition ${(hoveredCell == null ? void 0 : hoveredCell.c) === cIdx ? "bg-indigo-100 text-indigo-700" : "text-slate-600"}`
      },
      tok
    )), ATTN_TOKENS.map((rowTok, rIdx) => /* @__PURE__ */ import_react.default.createElement(import_react.default.Fragment, { key: `row-${rowTok}` }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setActiveRow(rIdx),
        onMouseEnter: () => setActiveRow(rIdx),
        className: `flex items-center justify-end pr-2.5 font-bold rounded-md transition cursor-pointer ${activeRow === rIdx ? "bg-indigo-600 text-white shadow-2xs" : "text-slate-700 hover:bg-slate-200/70"}`
      },
      rowTok
    ), matrix[rIdx].map((val, cIdx) => {
      const isMasked = val === null;
      const isRowSelected = activeRow === rIdx;
      const isCellHovered = (hoveredCell == null ? void 0 : hoveredCell.r) === rIdx && (hoveredCell == null ? void 0 : hoveredCell.c) === cIdx;
      return /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          key: `${rIdx}-${cIdx}`,
          onMouseEnter: () => {
            setActiveRow(rIdx);
            setHoveredCell({ r: rIdx, c: cIdx });
          },
          style: {
            backgroundColor: isMasked ? "#f1f5f9" : `rgba(79, 70, 229, ${Math.max(0.12, val * 0.95)})`
          },
          className: `h-12 rounded-lg flex flex-col items-center justify-center border transition-all duration-200 cursor-pointer ${isCellHovered ? "ring-2 ring-indigo-600 scale-105 z-10 border-white" : isRowSelected ? "border-indigo-400" : "border-slate-200/60"} ${isMasked ? "text-slate-400" : val > 0.45 ? "text-white font-bold" : "text-slate-800 font-medium"}`
        },
        /* @__PURE__ */ import_react.default.createElement("span", null, isMasked ? "-inf" : val.toFixed(2)),
        isMasked && /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[9px] font-sans" }, "masked")
      );
    }))))), /* @__PURE__ */ import_react.default.createElement("div", { className: "lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between border-b border-slate-200 pb-2.5" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] uppercase tracking-wider text-slate-400 font-semibold" }, "Query Token Focus"), /* @__PURE__ */ import_react.default.createElement("h4", { className: "text-sm font-bold text-[#17394D]" }, 'Token "', ATTN_TOKENS[activeRow], '" attends to:')), /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-1 rounded bg-indigo-100 text-indigo-700 font-mono text-xs font-bold" }, "Pos #", activeRow)), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2.5" }, ATTN_TOKENS.map((targetTok, idx) => {
      const weight = activeWeights[idx];
      const pct = weight === null ? 0 : Math.round(weight * 100);
      return /* @__PURE__ */ import_react.default.createElement("div", { key: targetTok, className: "space-y-1" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between text-xs font-mono" }, /* @__PURE__ */ import_react.default.createElement("span", { className: weight === null ? "text-slate-400 line-through" : "font-semibold text-slate-700" }, '\u2192 "', targetTok, '"'), /* @__PURE__ */ import_react.default.createElement("span", { className: weight === null ? "text-slate-400" : "font-bold text-indigo-600" }, weight === null ? "0% (-inf masked)" : `${pct}%`)), /* @__PURE__ */ import_react.default.createElement("div", { className: "w-full h-2 bg-slate-200 rounded-full overflow-hidden" }, /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          className: "h-full bg-indigo-600 rounded-full transition-all duration-300",
          style: { width: `${pct}%` }
        }
      )));
    })), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-[11px] text-slate-500 leading-relaxed pt-1 border-t border-slate-200/80" }, causalMask ? "With Causal Masking ON, tokens cannot attend to future positions (upper triangle = -inf), preventing data leakage during autoregressive training." : "With Causal Masking OFF (Encoder mode like BERT), every token attends bidirectionally to all tokens in the sequence."))));
  }

function XYFlowArchitectureLab() {
    const [selectedNodeId, setSelectedNodeId] = (0, import_react.useState)("mha");
    const selectedNode = TRANSFORMER_FLOW_NODES.find((n) => n.id === selectedNodeId) || TRANSFORMER_FLOW_NODES[1];
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-700 rounded-md" }, "Interactive Lab \u2022 @xyflow/react"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-extrabold text-[#17394D]" }, "Transformer Block Architecture Flow")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Click any node in the flow graph (including the curved Residual +x highway) to inspect tensor shapes & math.")), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-mono bg-slate-100 text-slate-700 px-3 py-1 rounded-lg border border-slate-200" }, "GPT-2 Small Block (N = 12 Layers)")), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch" }, /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        className: "lg:col-span-7 relative min-h-[395px] rounded-xl border border-slate-200 overflow-hidden select-none",
        style: {
          backgroundColor: "#f8fafc",
          backgroundImage: "radial-gradient(#cbd5e1 1px, transparent 1px)",
          backgroundSize: "18px 18px"
        }
      },
      /* @__PURE__ */ import_react.default.createElement("svg", { viewBox: "0 0 460 395", className: "w-full h-full" }, /* @__PURE__ */ import_react.default.createElement("defs", null, /* @__PURE__ */ import_react.default.createElement("marker", { id: "arrowIndigo", viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }, /* @__PURE__ */ import_react.default.createElement("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: "#4f46e5" })), /* @__PURE__ */ import_react.default.createElement("marker", { id: "arrowEmerald", viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }, /* @__PURE__ */ import_react.default.createElement("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: "#059669" }))), /* @__PURE__ */ import_react.default.createElement("line", { x1: "200", y1: "82", x2: "200", y2: "114", stroke: "#4f46e5", strokeWidth: "2.2", strokeDasharray: "5 5", markerEnd: "url(#arrowIndigo)" }), /* @__PURE__ */ import_react.default.createElement("line", { x1: "200", y1: "176", x2: "200", y2: "208", stroke: "#4f46e5", strokeWidth: "2.2", strokeDasharray: "5 5", markerEnd: "url(#arrowIndigo)" }), /* @__PURE__ */ import_react.default.createElement("line", { x1: "200", y1: "270", x2: "200", y2: "302", stroke: "#4f46e5", strokeWidth: "2.2", strokeDasharray: "5 5", markerEnd: "url(#arrowIndigo)" }), /* @__PURE__ */ import_react.default.createElement(
        "path",
        {
          d: "M 355 54 C 430 54, 430 242, 358 242",
          fill: "none",
          stroke: "#059669",
          strokeWidth: "2.2",
          strokeDasharray: "6 4",
          markerEnd: "url(#arrowEmerald)"
        }
      ), /* @__PURE__ */ import_react.default.createElement("rect", { x: "366", y: "136", width: "84", height: "22", rx: "6", fill: "#ecfdf5", stroke: "#6ee7b7" }), /* @__PURE__ */ import_react.default.createElement("text", { x: "408", y: "150", textAnchor: "middle", fontSize: "10", fontWeight: "700", fill: "#047857", fontFamily: "monospace" }, "Residual +x")),
      /* @__PURE__ */ import_react.default.createElement("div", { className: "absolute inset-0 pointer-events-none" }, TRANSFORMER_FLOW_NODES.map((node) => {
        const isSelected = node.id === selectedNodeId;
        return /* @__PURE__ */ import_react.default.createElement(
          "button",
          {
            key: node.id,
            type: "button",
            onClick: () => setSelectedNodeId(node.id),
            style: { top: `${node.y}px`, left: "24px", width: "326px" },
            className: `pointer-events-auto absolute text-left px-3.5 py-2.5 rounded-xl border-2 transition-all cursor-pointer shadow-2xs ${node.accent} ${isSelected ? "ring-2 ring-indigo-500/30 border-indigo-600 scale-[1.02] shadow-md" : "hover:border-slate-400"}`
          },
          /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold text-[#17394D] truncate" }, node.title), /* @__PURE__ */ import_react.default.createElement("span", { className: `px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold shrink-0 ${node.badge}` }, node.shape)),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] text-slate-500 mt-0.5" }, node.subtitle)
        );
      }))
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-5 flex flex-col justify-between space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-3" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-indigo-100 text-indigo-700" }, "Layer Inspector"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-mono text-slate-400" }, "d_model = 768")), /* @__PURE__ */ import_react.default.createElement("h4", { className: "text-base font-bold text-[#17394D]" }, selectedNode.title), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-600 leading-relaxed" }, selectedNode.details.explanation), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2 pt-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3 rounded-lg bg-white border border-slate-200" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[10px] font-semibold uppercase tracking-wider text-slate-400" }, "Tensor Transformation"), /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs font-bold text-indigo-700 mt-0.5" }, selectedNode.shape)), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3 rounded-lg bg-white border border-slate-200" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[10px] font-semibold uppercase tracking-wider text-slate-400" }, "Mathematical Operation"), /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs font-semibold text-slate-800 mt-0.5" }, selectedNode.details.operation)), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3 rounded-lg bg-white border border-slate-200" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[10px] font-semibold uppercase tracking-wider text-slate-400" }, "Learnable Parameters"), /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs text-emerald-700 font-semibold mt-0.5" }, selectedNode.details.params)))), /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] text-slate-400 flex items-center justify-between pt-2 border-t border-slate-200" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Batch (B) = 1, Seq Len (T) = 4"), /* @__PURE__ */ import_react.default.createElement("span", null, "Pre-Norm GPT Block")))));
  }

function RechartsTrainingSamplingLab() {
    const [temperature, setTemperature] = (0, import_react.useState)(0.7);
    const [topP, setTopP] = (0, import_react.useState)(0.9);
    const distribution = (0, import_react.useMemo)(() => {
      const scaledExps = CANDIDATE_LOGITS.map((item) => Math.exp(item.logit / temperature));
      const sumExp = scaledExps.reduce((acc, v) => acc + v, 0);
      let cumulativeProb = 0;
      return CANDIDATE_LOGITS.map((item, idx) => {
        const prob = scaledExps[idx] / sumExp;
        const prevCum = cumulativeProb;
        cumulativeProb += prob;
        const inNucleus = idx === 0 || prevCum < topP;
        return {
          token: item.token,
          logit: item.logit,
          prob,
          pct: Number((prob * 100).toFixed(1)),
          inNucleus
        };
      });
    }, [temperature, topP]);
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md" }, "Interactive Lab \u2022 recharts"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-extrabold text-[#17394D]" }, "Temperature & Nucleus (Top-P) Sampling")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Formula: ", /* @__PURE__ */ import_react.default.createElement("code", { className: "font-mono text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded" }, "P(x\u1D62) = exp(z\u1D62 / T) / \u03A3 exp(z\u2C7C / T)"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "px-3.5 py-1.5 rounded-xl bg-slate-900 text-slate-100 font-mono text-xs flex items-center gap-1.5 shadow-2xs" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-slate-400" }, "Prompt:"), /* @__PURE__ */ import_react.default.createElement("span", null, '"The capital of France is'), /* @__PURE__ */ import_react.default.createElement("span", { className: "px-1.5 py-0.5 rounded bg-indigo-500 text-white font-bold animate-pulse" }, "___"), /* @__PURE__ */ import_react.default.createElement("span", null, '"'))), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-5" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between items-center" }, /* @__PURE__ */ import_react.default.createElement("label", { className: "text-xs font-bold text-[#17394D]" }, "Temperature (T)"), /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 rounded-md bg-indigo-100 text-indigo-700 font-mono text-xs font-bold" }, temperature.toFixed(1))), /* @__PURE__ */ import_react.default.createElement(
      "input",
      {
        type: "range",
        min: "0.2",
        max: "2.0",
        step: "0.1",
        value: temperature,
        onChange: (e) => setTemperature(parseFloat(e.target.value)),
        className: "w-full accent-indigo-600 cursor-pointer"
      }
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between text-[10px] text-slate-500" }, /* @__PURE__ */ import_react.default.createElement("span", null, "0.2 (Deterministic)"), /* @__PURE__ */ import_react.default.createElement("span", null, "1.0 (Balanced)"), /* @__PURE__ */ import_react.default.createElement("span", null, "2.0 (High Entropy)"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2 pt-3 border-t border-slate-200" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between items-center" }, /* @__PURE__ */ import_react.default.createElement("label", { className: "text-xs font-bold text-[#17394D]" }, "Top-P (Nucleus Cutoff)"), /* @__PURE__ */ import_react.default.createElement("span", { className: "px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-700 font-mono text-xs font-bold" }, topP.toFixed(2))), /* @__PURE__ */ import_react.default.createElement(
      "input",
      {
        type: "range",
        min: "0.10",
        max: "1.00",
        step: "0.05",
        value: topP,
        onChange: (e) => setTopP(parseFloat(e.target.value)),
        className: "w-full accent-emerald-600 cursor-pointer"
      }
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between text-[10px] text-slate-500" }, /* @__PURE__ */ import_react.default.createElement("span", null, "0.10 (Strict)"), /* @__PURE__ */ import_react.default.createElement("span", null, "0.90 (Standard)"), /* @__PURE__ */ import_react.default.createElement("span", null, "1.00 (All Tokens)"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex gap-2 pt-1" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          setTemperature(0.2);
          setTopP(0.5);
        },
        className: "flex-1 py-1.5 text-[11px] font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-100 transition"
      },
      "Factual Coding"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          setTemperature(1.5);
          setTopP(0.95);
        },
        className: "flex-1 py-1.5 text-[11px] font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-100 transition"
      },
      "Creative Writing"
    ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "lg:col-span-7 space-y-3" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between text-xs text-slate-500 px-1" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-semibold text-[#17394D]" }, "Next-Token Softmax Probability Distribution"), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-3 text-[11px]" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "flex items-center gap-1" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-xs bg-indigo-600" }), " Active Nucleus"), /* @__PURE__ */ import_react.default.createElement("span", { className: "flex items-center gap-1" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-xs bg-slate-300" }), " Filtered Out"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3" }, distribution.map((row) => /* @__PURE__ */ import_react.default.createElement("div", { key: row.token, className: "space-y-1" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between text-xs font-mono" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement(
      "span",
      {
        className: `font-bold px-2 py-0.5 rounded ${row.inNucleus ? "bg-white text-slate-900 border border-slate-200" : "text-slate-400 line-through"}`
      },
      '"',
      row.token,
      '"'
    ), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] text-slate-400" }, "logit: ", row.logit.toFixed(1))), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, !row.inNucleus && /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] font-sans text-slate-400 bg-slate-200/70 px-1.5 py-0.5 rounded" }, "Cut by Top-P"), /* @__PURE__ */ import_react.default.createElement("span", { className: `font-bold w-12 text-right ${row.inNucleus ? "text-indigo-700" : "text-slate-400"}` }, row.pct, "%"))), /* @__PURE__ */ import_react.default.createElement("div", { className: "w-full h-5 bg-slate-200/70 rounded-lg overflow-hidden p-0.5" }, /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        style: { width: `${Math.max(2, row.pct)}%` },
        className: `h-full rounded-md transition-all duration-300 ${row.inNucleus ? row.token === "Paris" ? "bg-indigo-600" : "bg-emerald-500" : "bg-slate-300"}`
      }
    ))))))));
  }

function ConceptAnimationLab({ animType }) {
    const [step, setStep] = (0, import_react.useState)(0);
    const [playing, setPlaying] = (0, import_react.useState)(true);
    (0, import_react.useEffect)(() => {
      setStep(0);
      setPlaying(true);
    }, [animType]);
    (0, import_react.useEffect)(() => {
      if (!playing)
        return;
      const timer = setInterval(() => {
        setStep((s) => (s + 1) % 4);
      }, 1700);
      return () => clearInterval(timer);
    }, [playing]);
    if (animType === "bpe_merge")
      return /* @__PURE__ */ import_react.default.createElement(MotionBPELab, null);
    if (animType === "embeddings_pos")
      return /* @__PURE__ */ import_react.default.createElement(ThreeDEmbeddingLab, null);
    if (animType === "qkv_attention" || animType === "causal_mask")
      return /* @__PURE__ */ import_react.default.createElement(VisxAttentionHeatmapLab, null);
    if (animType === "transformer_block")
      return /* @__PURE__ */ import_react.default.createElement(XYFlowArchitectureLab, null);
    if (animType === "lr_cosine" || animType === "softmax_logits")
      return /* @__PURE__ */ import_react.default.createElement(RechartsTrainingSamplingLab, null);
    const controls = (label) => /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-200/80" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2.5 h-2.5 rounded-full bg-teal-500 animate-ping" }), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-extrabold uppercase tracking-wider text-[#17394D]" }, label)), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setPlaying((p) => !p),
        className: "text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-[#17394D]"
      },
      playing ? "Pause" : "Auto-Play"
    ), [0, 1, 2, 3].map((idx) => /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        key: idx,
        type: "button",
        onClick: () => {
          setPlaying(false);
          setStep(idx);
        },
        className: `w-6 h-6 rounded-md text-[11px] font-mono font-bold transition-all ${step === idx ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`
      },
      idx + 1
    ))));
    if (animType === "python_oop") {
      const steps = [
        { title: "Step 1: Isolated .venv Sandbox", badge: "py -3.11 -m venv .venv", desc: "Creates a private Python folder so PyTorch & Transformers never conflict with system apps." },
        { title: "Step 2: @dataclass Config Blueprint", badge: "cfg = GPTConfig(emb_dim=384, n_heads=6)", desc: "Bundles all architecture knobs into one clean, readable configuration object." },
        { title: "Step 3: Token Dict Lookup (O(1))", badge: 'vocab["Paris"] -> 6342', desc: "Python dictionaries map words to integer IDs instantaneously." },
        { title: "Step 4: Stateful nn.Module Class", badge: "model = GPTModel(cfg)", desc: "Instantiates every layer and registers learnable weight matrices automatically." }
      ];
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Mental Model: Python OOP & .venv Architecture"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-4 gap-2.5" }, steps.map((s, idx) => /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          key: idx,
          className: `p-3 rounded-xl border transition-all duration-300 ${step === idx ? "bg-gradient-to-br from-teal-50 to-indigo-50 border-indigo-500 scale-[1.02] shadow-xs" : "bg-slate-50/70 border-slate-200/80 opacity-65"}`
        },
        /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-bold text-[#17394D]" }, s.title),
        /* @__PURE__ */ import_react.default.createElement("code", { className: "block text-[10px] font-mono bg-white border border-slate-200 rounded px-1.5 py-1 my-1.5 text-indigo-700 truncate" }, s.badge),
        /* @__PURE__ */ import_react.default.createElement("p", { className: "text-[11px] text-slate-600 leading-snug" }, s.desc)
      ))));
    }
    if (animType === "broadcasting") {
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Visual: NumPy Broadcasting (4\xD73 Matrix + 1\xD73 Bias)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col sm:flex-row items-center justify-center gap-4 py-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3 rounded-xl bg-slate-50 border border-slate-200 text-center" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-mono font-bold text-slate-500 mb-1.5" }, "Token Matrix X (4 \xD7 3)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-3 gap-1 font-mono text-xs" }, [1, 0.2, -0.5, 0.9, 0.1, -0.4, -0.8, 0.7, 0.3, 0, -0.5, 0.9].map((v, i) => /* @__PURE__ */ import_react.default.createElement("span", { key: i, className: "px-2 py-1 rounded bg-white border border-slate-200 text-[#17394D]" }, v.toFixed(1))))), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xl font-extrabold text-indigo-600" }, "+"), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3 rounded-xl bg-teal-50/70 border border-teal-300 text-center" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-mono font-bold text-teal-800 mb-1.5" }, "Bias (3,) \u2192 Stretched to (4 \xD7 3)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-1 font-mono text-xs" }, [0, 1, 2, 3].map((row) => /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          key: row,
          className: `grid grid-cols-3 gap-1 transition-all duration-300 ${row === 0 || step >= 1 ? "opacity-100" : "opacity-25 scale-95"}`
        },
        [0.1, -0.1, 0.5].map((b, c) => /* @__PURE__ */ import_react.default.createElement(
          "span",
          {
            key: c,
            className: `px-2 py-1 rounded border ${row === 0 ? "bg-teal-600 text-white border-teal-700 font-bold" : "bg-teal-100/80 text-teal-900 border-teal-300 border-dashed"}`
          },
          b.toFixed(1)
        ))
      ))))), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-center text-slate-600 mt-2" }, step === 0 ? "1D bias vector starts as a single row of 3 numbers." : "NumPy automatically broadcasts that single row across all 4 token rows without copying memory!"));
    }
    if (animType === "tensor_btc") {
      const stages = [
        { label: "Input Token IDs", shape: "(B=2, T=5)", note: "2 sentences, 5 word IDs each" },
        { label: "After Embedding", shape: "(B=2, T=5, C=64)", note: "Each token becomes a 64-float vector" },
        { label: "Split 4 Heads", shape: "(B=2, H=4, T=5, d=16)", note: "64 dims split into 4 parallel spotlights" },
        { label: "LM Head Logits", shape: "(B=2, T=5, V=50257)", note: "Scores every word in vocabulary!" }
      ];
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Tensor Shape Tracer: (Batch, Tokens, Channels)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-2 sm:grid-cols-4 gap-3" }, stages.map((st, idx) => /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          key: idx,
          className: `p-3.5 rounded-xl border transition-all duration-300 ${step === idx ? "bg-indigo-600 text-white border-indigo-700 shadow-sm scale-[1.03]" : "bg-slate-50 text-[#17394D] border-slate-200"}`
        },
        /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-bold opacity-85" }, st.label),
        /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs font-extrabold my-1.5" }, st.shape),
        /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] opacity-85 leading-snug" }, st.note)
      ))));
    }
    if (animType === "autograd_graph") {
      const nodes = [
        { id: "w=3, x=2", val: "z = 7.0", grad: "dL/dw = -12.0" },
        { id: "target y=10", val: "diff = -3.0", grad: "dL/ddiff = -6.0" },
        { id: "Loss = diff\xB2", val: "Loss = 9.0", grad: "dL/dL = 1.0" }
      ];
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Backpropagation Flow: Forward Pass vs. Backward Gradients"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-3 gap-3" }, nodes.map((n, idx) => {
        const isForward = step <= 1 && step === idx;
        const isBackward = step >= 2 && 3 - step === idx;
        return /* @__PURE__ */ import_react.default.createElement(
          "div",
          {
            key: n.id,
            className: `p-3.5 rounded-xl border transition-all duration-300 ${isBackward ? "bg-rose-50 border-rose-500 shadow-xs" : isForward ? "bg-teal-50 border-teal-500 shadow-xs" : "bg-slate-50 border-slate-200"}`
          },
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-mono text-slate-500" }, n.id),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-sm font-extrabold text-[#17394D] my-1" }, n.val),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-xs font-mono font-bold text-rose-700" }, n.grad)
        );
      })), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-600 mt-3 text-center" }, step <= 1 ? "Forward Pass (left -> right): computing prediction z=7.0 and squared error Loss=9.0." : "Backward Pass (right -> left): Chain Rule multiplies gradients backward so weight w knows to increase!"));
    }
    if (animType === "gradient_descent") {
      const points = [
        { stepLabel: "Step 0 (Random)", w: -3.5, loss: 12.25 },
        { stepLabel: "Step 25", w: -1.8, loss: 3.24 },
        { stepLabel: "Step 60", w: -0.6, loss: 0.36 },
        { stepLabel: "Step 120 (Converged)", w: 0, loss: 0.01 }
      ];
      const cur = points[step];
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Optimization: Stepping Down the Loss Valley"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-2 sm:grid-cols-4 gap-2.5" }, points.map((pt, idx) => /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          key: pt.stepLabel,
          className: `p-3 rounded-xl border transition-all ${step === idx ? "bg-amber-50 border-amber-500 shadow-xs" : "bg-slate-50 border-slate-200 opacity-70"}`
        },
        /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-bold text-[#17394D]" }, pt.stepLabel),
        /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs text-indigo-700 mt-1" }, "Weight offset: ", pt.w),
        /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-xs font-extrabold text-amber-800" }, "Loss: ", pt.loss)
      ))), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-center text-slate-600 mt-2.5" }, "Current State: ", /* @__PURE__ */ import_react.default.createElement("strong", null, cur.stepLabel), " \u2014 AdamW nudges weights downhill until Loss approaches 0."));
    }
    if (animType === "sliding_window") {
      const stream = ["The", "capital", "of", "France", "is", "Paris", "<|eos|>"];
      const start = Math.min(step, 2);
      const xSlice = stream.slice(start, start + 4);
      const ySlice = stream.slice(start + 1, start + 5);
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Sliding-Window Next-Token Dataset (X vs. Shifted Y)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2.5 font-mono text-xs" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-24 font-bold text-indigo-700" }, "Input X:"), xSlice.map((t, i) => /* @__PURE__ */ import_react.default.createElement("span", { key: i, className: "px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-300 text-indigo-900 font-bold" }, t))), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-24 font-bold text-teal-700" }, "Target Y (+1):"), ySlice.map((t, i) => /* @__PURE__ */ import_react.default.createElement("span", { key: i, className: "px-2.5 py-1 rounded-lg bg-teal-50 border border-teal-300 text-teal-900 font-bold" }, t)))));
    }
    if (animType === "lora_lowrank") {
      return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated LoRA Adapter: Frozen Brain + Tiny Trainable A @ B Matrices"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3.5 rounded-xl bg-sky-50 border border-sky-200" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "font-extrabold text-sky-950" }, "Base Pretrained Matrix W (4096 \xD7 4096)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-sky-800 mt-1" }, "16,777,216 params \u2022 FROZEN (requires_grad=False)")), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-3.5 rounded-xl bg-emerald-50 border border-emerald-300" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "font-extrabold text-emerald-950" }, "LoRA A (4096\xD78) @ B (8\xD74096)"), /* @__PURE__ */ import_react.default.createElement("div", { className: "font-mono text-emerald-800 mt-1" }, "65,536 params (0.39%) \u2022 TRAINABLE!"))));
    }
    const hops = [
      "1. Client sends JSON: {'prompt': 'Explain DNS'}",
      "2. Pydantic validates types & FastAPI routes request",
      "3. PyTorch GPT runs inside torch.no_grad() on GPU/CPU",
      "4. Uvicorn returns JSON response in milliseconds!"
    ];
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs" }, controls("Animated Local LLM Inference Request Lifecycle"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 gap-2.5" }, hops.map((h, idx) => /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        key: h,
        className: `p-3 rounded-xl border text-xs font-bold transition-all ${step === idx ? "bg-teal-600 text-white border-teal-700" : "bg-slate-50 text-[#17394D] border-slate-200"}`
      },
      h
    ))));
  }

function App() {
    const [completedIds, setCompletedIds] = (0, import_react.useState)(() => {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        const parsed = saved ? JSON.parse(saved) : [];
        return new Set(Array.isArray(parsed) ? parsed : []);
      } catch (e) {
        return /* @__PURE__ */ new Set();
      }
    });
    const [currentFilter, setCurrentFilter] = (0, import_react.useState)("all");
    const [collapsedPhases, setCollapsedPhases] = (0, import_react.useState)(() => /* @__PURE__ */ new Set());
    const [activeStageId, setActiveStageId] = (0, import_react.useState)(1);
    const [activeTopicId, setActiveTopicId] = (0, import_react.useState)(null);
    const [quizChoice, setQuizChoice] = (0, import_react.useState)(null);
    const [copiedCode, setCopiedCode] = (0, import_react.useState)(false);
    const [scrollSpinePct, setScrollSpinePct] = (0, import_react.useState)(0);
    const [toast, setToast] = (0, import_react.useState)(null);
    const [copiedSetup, setCopiedSetup] = (0, import_react.useState)(false);
    const timelineRef = (0, import_react.useRef)(null);
    (0, import_react.useEffect)(() => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify([...completedIds]));
      } catch (e) {
      }
    }, [completedIds]);
    (0, import_react.useEffect)(() => {
      setQuizChoice(null);
      setCopiedCode(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }, [activeTopicId]);
    (0, import_react.useEffect)(() => {
      const handleScroll = () => {
        const el = timelineRef.current;
        if (!el)
          return;
        const rect = el.getBoundingClientRect();
        const windowH = window.innerHeight;
        const totalScrollable = rect.height - windowH * 0.4;
        if (totalScrollable <= 0) {
          setScrollSpinePct(100);
          return;
        }
        const scrolled = Math.min(Math.max(windowH * 0.5 - rect.top, 0), totalScrollable);
        setScrollSpinePct(Math.round(scrolled / totalScrollable * 100));
      };
      handleScroll();
      window.addEventListener("scroll", handleScroll, { passive: true });
      window.addEventListener("resize", handleScroll, { passive: true });
      return () => {
        window.removeEventListener("scroll", handleScroll);
        window.removeEventListener("resize", handleScroll);
      };
    }, [activeTopicId]);
    const totalTopics = ALL_TOPICS.length;
    const doneCount = completedIds.size;
    const masteryPct = Math.round(doneCount / totalTopics * 100);
    const activeStage = PIPELINE_STAGES.find((s) => s.id === activeStageId) || PIPELINE_STAGES[0];
    const activeTopicIndex = (0, import_react.useMemo)(
      () => ALL_TOPICS.findIndex((t) => t.id === activeTopicId),
      [activeTopicId]
    );
    const activeTopic = activeTopicIndex >= 0 ? ALL_TOPICS[activeTopicIndex] : null;
    const filteredPhases = (0, import_react.useMemo)(() => {
      if (currentFilter === "all")
        return ROADMAP_PHASES;
      return ROADMAP_PHASES.filter((p) => p.id === currentFilter);
    }, [currentFilter]);
    const toggleTopic = (topicId) => {
      setCompletedIds((prev) => {
        const next = new Set(prev);
        if (next.has(topicId))
          next.delete(topicId);
        else
          next.add(topicId);
        return next;
      });
    };
    const togglePhaseCollapse = (phaseId) => {
      setCollapsedPhases((prev) => {
        const next = new Set(prev);
        if (next.has(phaseId))
          next.delete(phaseId);
        else
          next.add(phaseId);
        return next;
      });
    };
    const handleJumpToPhase = (phaseId) => {
      setActiveTopicId(null);
      if (currentFilter !== "all")
        setCurrentFilter("all");
      setCollapsedPhases((prev) => {
        const next = new Set(prev);
        next.delete(phaseId);
        return next;
      });
      setTimeout(() => {
        const target = document.getElementById(phaseId);
        if (target)
          target.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    };
    return /* @__PURE__ */ import_react.default.createElement("div", { className: "min-h-screen bg-[#F4F7FA] text-[#17394D] font-sans antialiased selection:bg-teal-100 selection:text-teal-900" }, /* @__PURE__ */ import_react.default.createElement("header", { className: "sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200/80 transition-all" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setActiveTopicId(null),
        className: "flex items-center gap-2.5 text-left group"
      },
      /* @__PURE__ */ import_react.default.createElement("div", { className: "w-9 h-9 rounded-xl bg-gradient-to-br from-teal-600 to-indigo-600 flex items-center justify-center text-white shadow-sm" }, /* @__PURE__ */ import_react.default.createElement("svg", { className: "w-5 h-5", fill: "none", stroke: "currentColor", strokeWidth: "2", viewBox: "0 0 24 24" }, /* @__PURE__ */ import_react.default.createElement("circle", { cx: "6", cy: "6", r: "2.5" }), /* @__PURE__ */ import_react.default.createElement("circle", { cx: "18", cy: "6", r: "2.5" }), /* @__PURE__ */ import_react.default.createElement("circle", { cx: "12", cy: "18", r: "2.5" }), /* @__PURE__ */ import_react.default.createElement("path", { strokeLinecap: "round", d: "M8.2 7.2L10.8 15.8M15.8 7.2L13.2 15.8M8.5 6h7" }))),
      /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-[#17394D] group-hover:text-indigo-700 tracking-tight text-sm sm:text-base block leading-none" }, "LLM From Scratch Academy"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-medium text-slate-500" }, activeTopic ? `Module ${activeTopic.codeNum} Lesson Page` : "Self-Contained Interactive Course"))
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2.5 sm:gap-3 flex-1 max-w-xl justify-end" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex-1 hidden md:block max-w-[200px]" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between text-xs font-semibold mb-1" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-slate-600" }, "Mastery"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-teal-700 font-mono" }, masteryPct, "% (", doneCount, "/", totalTopics, ")")), /* @__PURE__ */ import_react.default.createElement("div", { className: "w-full h-2 bg-slate-200/80 rounded-full overflow-hidden" }, /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        className: "h-full bg-gradient-to-r from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-500 ease-out",
        style: { width: `${masteryPct}%` }
      }
    ))), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          downloadViteProjectZip();
          setToast({ message: "Downloaded llm-from-scratch-academy-vite.zip!", confirm: false });
          setTimeout(() => setToast(null), 3200);
        },
        className: "inline-flex items-center gap-1.5 text-xs font-bold text-white bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 px-3 py-2 rounded-xl shadow-xs transition-all shrink-0"
      },
      /* @__PURE__ */ import_react.default.createElement("span", null, "Download Vite Project (.zip)")
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setToast({ message: "Reset all saved roadmap checkmarks?", confirm: true }),
        title: "Reset saved progress",
        className: "text-xs font-medium text-slate-500 hover:text-rose-600 px-2 py-1.5 rounded-lg hover:bg-rose-50 transition-colors"
      },
      "Reset"
    )))), activeTopic ? /* @__PURE__ */ import_react.default.createElement("main", { className: "max-w-5xl mx-auto px-4 sm:px-6 py-8 pb-24 space-y-6" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-3" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setActiveTopicId(null),
        className: "inline-flex items-center gap-2 text-xs sm:text-sm font-bold text-[#17394D] hover:text-indigo-700 bg-white border border-slate-200 px-4 py-2 rounded-xl shadow-2xs transition-all"
      },
      /* @__PURE__ */ import_react.default.createElement("span", null, "\u2190 Back to All 6 Phases")
    ), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2 text-xs font-mono text-slate-500" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Phase ", activeTopic.phaseNumber), /* @__PURE__ */ import_react.default.createElement("span", null, "/"), /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-indigo-700" }, "Module ", activeTopic.codeNum), /* @__PURE__ */ import_react.default.createElement("span", null, "\u2022"), /* @__PURE__ */ import_react.default.createElement("span", null, activeTopic.duration))), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center justify-between gap-3" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold uppercase tracking-wider px-3 py-1 rounded-full bg-teal-50 text-teal-800 border border-teal-200" }, "Phase ", activeTopic.phaseNumber, ": ", activeTopic.phaseTitle), /* @__PURE__ */ import_react.default.createElement("label", { className: "flex items-center gap-2 cursor-pointer select-none px-3.5 py-1.5 rounded-xl bg-slate-50 border border-slate-200 hover:border-teal-400" }, /* @__PURE__ */ import_react.default.createElement(
      "input",
      {
        type: "checkbox",
        checked: completedIds.has(activeTopic.id),
        onChange: () => toggleTopic(activeTopic.id),
        className: "w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
      }
    ), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold text-[#17394D]" }, completedIds.has(activeTopic.id) ? "Completed \u2713" : "Mark Module Completed"))), /* @__PURE__ */ import_react.default.createElement("h1", { className: "text-2xl sm:text-4xl font-extrabold text-[#17394D] tracking-tight" }, activeTopic.codeNum, " \u2014 ", activeTopic.title), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-sm sm:text-base text-slate-700 leading-relaxed bg-sky-50/75 border border-sky-200/80 rounded-xl p-4" }, /* @__PURE__ */ import_react.default.createElement("strong", { className: "text-sky-950" }, "Plain-English Definition: "), activeTopic.plainEnglish)), /* @__PURE__ */ import_react.default.createElement(ConceptAnimationLab, { animType: activeTopic.animType }), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6" }, /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-3" }, "Key Terms & Jargon You Will Master Here"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 gap-2.5" }, activeTopic.jargonTerms.map((j) => /* @__PURE__ */ import_react.default.createElement("div", { key: j.term, className: "p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-xs" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-mono font-bold text-indigo-700 block mb-0.5" }, j.term), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-slate-700 leading-snug" }, j.def))))), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-5" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "border-b border-slate-100 pb-3" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-extrabold uppercase tracking-wider text-teal-700" }, "Complete In-App Lesson Guide"), /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-xl font-extrabold text-[#17394D] mt-0.5" }, "Core Concepts & Step-by-Step Intuition")), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-teal-50/60 border border-teal-200/80 rounded-xl p-4 text-sm text-[#17394D]" }, /* @__PURE__ */ import_react.default.createElement("strong", { className: "text-teal-950" }, "Why It Matters for Building LLMs: "), activeTopic.why), (activeTopic.lessonSections || []).map((sec, idx) => /* @__PURE__ */ import_react.default.createElement("div", { key: idx, className: "space-y-2.5 pt-2 border-t border-slate-100 first:border-t-0 first:pt-0" }, /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-bold text-[#17394D]" }, sec.heading), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-sm text-slate-700 leading-relaxed" }, sec.body), sec.codeExample && /* @__PURE__ */ import_react.default.createElement("pre", { className: "bg-slate-900 text-slate-100 rounded-xl p-4 text-xs font-mono overflow-x-auto leading-relaxed border border-slate-800" }, /* @__PURE__ */ import_react.default.createElement("code", null, sec.codeExample)))), /* @__PURE__ */ import_react.default.createElement("div", { className: "pt-2 border-t border-slate-100" }, /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-2.5" }, "Learning Checklist for Module ", activeTopic.codeNum), /* @__PURE__ */ import_react.default.createElement("ul", { className: "space-y-2" }, activeTopic.concepts.map((c, idx) => /* @__PURE__ */ import_react.default.createElement("li", { key: idx, className: "flex items-start gap-2.5 text-sm text-[#17394D]" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-5 h-5 rounded-full bg-teal-50 text-teal-700 border border-teal-200 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5" }, "\u2713"), /* @__PURE__ */ import_react.default.createElement("span", null, c)))))), activeTopic.liveExample && /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-gradient-to-br from-violet-50/90 to-indigo-50/60 border border-violet-200/90 rounded-2xl p-6 shadow-2xs" }, /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-xs font-extrabold uppercase tracking-wider text-violet-950 mb-1.5 flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2 h-2 rounded-full bg-violet-600" }), "Where Live Running AIs Use This Exact Concept"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-sm text-[#17394D] leading-relaxed" }, activeTopic.liveExample)), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-slate-900 rounded-2xl border border-slate-800 p-6 text-slate-100 space-y-3" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-mono font-bold text-teal-400" }, "Hands-On Python 3.11 / PyTorch Reference Code"), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          var _a;
          (_a = navigator.clipboard) == null ? void 0 : _a.writeText(activeTopic.code);
          setCopiedCode(true);
          setTimeout(() => setCopiedCode(false), 2e3);
        },
        className: "text-xs font-semibold px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
      },
      copiedCode ? "Copied!" : "Copy Code"
    )), /* @__PURE__ */ import_react.default.createElement("pre", { className: "text-xs sm:text-sm font-mono overflow-x-auto leading-relaxed" }, /* @__PURE__ */ import_react.default.createElement("code", null, activeTopic.code))), activeTopic.quiz && /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-indigo-200/90 shadow-sm p-6 space-y-3" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-extrabold uppercase tracking-wider text-indigo-700" }, "Quick Knowledge Check"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-base font-bold text-[#17394D]" }, activeTopic.quiz.question), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-2" }, activeTopic.quiz.options.map((opt, idx) => {
      const chosen = quizChoice === idx;
      const isCorrect = idx === activeTopic.quiz.answerIndex;
      return /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          key: idx,
          type: "button",
          onClick: () => setQuizChoice(idx),
          className: `w-full text-left p-3.5 rounded-xl border text-xs sm:text-sm font-medium transition-all ${chosen ? isCorrect ? "bg-emerald-50 border-emerald-500 text-emerald-950 font-bold" : "bg-rose-50 border-rose-500 text-rose-950" : "bg-slate-50 hover:bg-white border-slate-200 text-[#17394D]"}`
        },
        opt
      );
    })), quizChoice !== null && /* @__PURE__ */ import_react.default.createElement(
      "div",
      {
        className: `p-3 rounded-xl text-xs leading-relaxed ${quizChoice === activeTopic.quiz.answerIndex ? "bg-emerald-50 text-emerald-900 border border-emerald-200" : "bg-amber-50 text-amber-900 border border-amber-200"}`
      },
      /* @__PURE__ */ import_react.default.createElement("strong", null, quizChoice === activeTopic.quiz.answerIndex ? "Correct! " : "Not quite \u2014 "),
      activeTopic.quiz.explanation
    )), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-rose-50/80 border border-rose-200/90 rounded-2xl p-5" }, /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-xs font-extrabold uppercase tracking-wider text-rose-800 mb-1.5" }, "Beginner Pitfall & Debugging Tip"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs sm:text-sm text-[#17394D] leading-relaxed" }, activeTopic.pitfall)), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-amber-50/80 border border-amber-200/90 rounded-2xl p-5" }, /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-xs font-extrabold uppercase tracking-wider text-amber-800 mb-1.5" }, "Hands-On Practice Build"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs sm:text-sm font-semibold text-[#17394D] leading-relaxed" }, activeTopic.project))), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 p-5 space-y-3" }, /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-xs font-extrabold uppercase tracking-wider text-slate-500" }, "Optional External Video & Blog References"), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 gap-2" }, activeTopic.links.map((link, idx) => /* @__PURE__ */ import_react.default.createElement(
      "a",
      {
        key: idx,
        href: link.url,
        target: "_blank",
        rel: "noopener noreferrer",
        className: "flex items-center justify-between gap-2 p-3 rounded-xl bg-slate-50 hover:bg-indigo-50/60 border border-slate-200 hover:border-indigo-300 transition-all"
      },
      /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-semibold text-[#17394D] truncate" }, link.label),
      /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold text-indigo-600 shrink-0" }, "\u2197")
    )))), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 p-4 flex flex-wrap items-center justify-between gap-3" }, /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        disabled: activeTopicIndex <= 0,
        onClick: () => activeTopicIndex > 0 && setActiveTopicId(ALL_TOPICS[activeTopicIndex - 1].id),
        className: "px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white text-xs sm:text-sm font-bold text-[#17394D] disabled:opacity-40"
      },
      "\u2190 Previous Module"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => toggleTopic(activeTopic.id),
        className: `px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${completedIds.has(activeTopic.id) ? "bg-emerald-600 text-white" : "bg-[#17394D] text-white hover:bg-indigo-900"}`
      },
      completedIds.has(activeTopic.id) ? "Completed \u2713" : "Mark Module Completed"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        disabled: activeTopicIndex >= ALL_TOPICS.length - 1,
        onClick: () => activeTopicIndex < ALL_TOPICS.length - 1 && setActiveTopicId(ALL_TOPICS[activeTopicIndex + 1].id),
        className: "px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white text-xs sm:text-sm font-bold text-[#17394D] disabled:opacity-40"
      },
      "Next Module \u2192"
    ))) : (
      /* =================================================================== */
      /* VIEW B: MAIN ROADMAP OVERVIEW + STEP 0 SETUP + 6 PHASE TIMELINE     */
      /* =================================================================== */
      /* @__PURE__ */ import_react.default.createElement(import_react.default.Fragment, null, /* @__PURE__ */ import_react.default.createElement("section", { className: "relative pt-10 pb-8 sm:pt-14 sm:pb-12 border-b border-slate-200/70" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "max-w-6xl mx-auto px-4 sm:px-6" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "max-w-3xl" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-teal-200/80 shadow-xs text-xs font-semibold text-teal-800 mb-4" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2 h-2 rounded-full bg-teal-500 animate-ping" }), /* @__PURE__ */ import_react.default.createElement("span", null, "All-In-One Interactive Course \u2022 31 Full Lesson Pages Included")), /* @__PURE__ */ import_react.default.createElement("h1", { className: "text-3xl sm:text-5xl font-extrabold text-[#17394D] tracking-tight leading-[1.15]" }, "Build Large Language Models", " ", /* @__PURE__ */ import_react.default.createElement("span", { className: "text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-indigo-600" }, "From Scratch")), /* @__PURE__ */ import_react.default.createElement("p", { className: "mt-4 text-base sm:text-lg text-slate-600 leading-relaxed" }, "Click any of the 31 modules below to open its full interactive lesson page\u2014complete with animated concept visualizers, plain-English explanations, PyTorch code walkthroughs, and self-check quizzes right inside this app.")), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-8 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-100" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2 h-2 rounded-full bg-indigo-500" }), "What You Will Build: End-to-End Decoder-Only LLM Pipeline"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Click any stage in the pipeline below to inspect how raw text becomes next-token predictions\u2014and jump to its phase.")), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2 text-xs font-mono text-slate-500 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200/70" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Input:"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-indigo-600 font-semibold" }, '"The capital of France is"'), /* @__PURE__ */ import_react.default.createElement("span", null, "\u2192"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-teal-700 font-semibold bg-teal-50 px-1.5 py-0.5 rounded" }, '"Paris"'))), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3" }, PIPELINE_STAGES.map((stage, idx) => {
        const isSelected = stage.id === activeStageId;
        return /* @__PURE__ */ import_react.default.createElement(
          "button",
          {
            key: stage.id,
            type: "button",
            onClick: () => setActiveStageId(stage.id),
            className: `group relative text-left p-3.5 rounded-xl border transition-all duration-200 cursor-pointer ${isSelected ? "bg-gradient-to-br from-teal-50/90 to-indigo-50/70 border-teal-500 shadow-xs ring-2 ring-teal-500/20" : "bg-slate-50/70 hover:bg-white border-slate-200/90 hover:border-indigo-300"}`
          },
          /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between mb-1.5" }, /* @__PURE__ */ import_react.default.createElement(
            "span",
            {
              className: `w-2.5 h-2.5 rounded-full ${isSelected ? "bg-teal-500" : "bg-slate-300 group-hover:bg-indigo-400"}`
            }
          ), idx < PIPELINE_STAGES.length - 1 ? /* @__PURE__ */ import_react.default.createElement("span", { className: "hidden lg:inline text-xs font-mono text-slate-400" }, "\u2192") : /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] font-mono text-teal-700 font-bold" }, "OUT")),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-xs font-bold text-[#17394D] group-hover:text-indigo-700 transition-colors" }, stage.name),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-mono text-slate-500 mt-0.5" }, stage.sub)
        );
      })), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-4 p-3.5 rounded-xl bg-gradient-to-r from-slate-50 to-indigo-50/40 border border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-all" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-start sm:items-center gap-3" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "shrink-0 px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-indigo-100 text-indigo-800" }, activeStage.badge), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs sm:text-sm text-[#17394D] font-medium" }, activeStage.desc)), /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          type: "button",
          onClick: () => handleJumpToPhase(activeStage.phaseId),
          className: "shrink-0 text-xs font-semibold text-teal-700 hover:text-teal-900 bg-white hover:bg-teal-50 border border-teal-200 px-3 py-1.5 rounded-lg shadow-2xs transition-all self-start sm:self-auto"
        },
        "Jump to Phase \u2193"
      ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-6 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6 space-y-5" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2 h-2 rounded-full bg-teal-500" }), "Step 0: Required Softwares, Python Packages & Setup Commands"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs text-slate-500 mt-0.5" }, "Install only what you need for building and running LLMs in Python & PyTorch (no Java/JDK/Maven needed).")), /* @__PURE__ */ import_react.default.createElement(
        "a",
        {
          href: "https://pytorch.org/get-started/locally/",
          target: "_blank",
          rel: "noopener noreferrer",
          className: "shrink-0 inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-700 hover:text-indigo-900 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 px-3 py-1.5 rounded-lg transition-colors self-start sm:self-auto"
        },
        /* @__PURE__ */ import_react.default.createElement("span", null, "PyTorch GPU/CPU Selector"),
        /* @__PURE__ */ import_react.default.createElement("span", null, "\u2197")
      )), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3" }, SETUP_SOFTWARES.map((sw) => /* @__PURE__ */ import_react.default.createElement(
        "a",
        {
          key: sw.name,
          href: sw.url,
          target: "_blank",
          rel: "noopener noreferrer",
          className: "group p-3.5 rounded-xl bg-slate-50/70 hover:bg-white border border-slate-200/90 hover:border-teal-400 transition-all flex flex-col justify-between gap-2.5"
        },
        /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-1.5" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-bold text-[#17394D] group-hover:text-teal-700" }, sw.name), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-bold text-teal-700 shrink-0" }, "Download \u2197")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-[11px] font-medium text-slate-700 leading-snug" }, sw.role), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-2 rounded-lg bg-violet-50/75 border border-violet-200/75 text-[10px] text-[#17394D] leading-snug" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-violet-900" }, "Where Used: "), sw.whereUsed), /* @__PURE__ */ import_react.default.createElement("div", { className: "p-2 rounded-lg bg-emerald-50/70 border border-emerald-200/75 text-[10px] text-[#17394D] leading-snug" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-emerald-900" }, "What You Build: "), sw.whatYouBuild)),
        /* @__PURE__ */ import_react.default.createElement("code", { className: "block text-[10px] font-mono bg-white border border-slate-200/80 rounded px-2 py-1 text-slate-600 truncate" }, sw.winget)
      ))), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "rounded-xl bg-sky-50/60 border border-sky-200/80 p-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-extrabold uppercase tracking-wider text-sky-950 block" }, "14 Python Packages (PyPI) \u2014 Purpose, Live AI Usage & What You Will Build")), /* @__PURE__ */ import_react.default.createElement(
        "a",
        {
          href: "https://pypi.org/",
          target: "_blank",
          rel: "noopener noreferrer",
          className: "text-[11px] font-semibold text-sky-800 hover:underline shrink-0"
        },
        "pypi.org \u2197"
      )), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5" }, SETUP_PACKAGES.map((pkg) => /* @__PURE__ */ import_react.default.createElement(
        "a",
        {
          key: pkg.name,
          href: `https://pypi.org/project/${pkg.name}/`,
          target: "_blank",
          rel: "noopener noreferrer",
          className: "group bg-white hover:bg-teal-50/40 border border-sky-200/90 hover:border-teal-400 rounded-xl p-3 text-[#17394D] transition-all flex flex-col justify-between gap-1.5"
        },
        /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("strong", { className: "font-mono text-xs text-indigo-700 group-hover:text-teal-700" }, pkg.name), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] font-semibold text-slate-400 group-hover:text-teal-700" }, "PyPI \u2197")), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-[11px] font-semibold text-[#17394D] mt-0.5 leading-snug" }, pkg.desc)),
        /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-1 pt-1.5 border-t border-slate-100 text-[10px] leading-snug" }, /* @__PURE__ */ import_react.default.createElement("p", { className: "text-slate-600" }, /* @__PURE__ */ import_react.default.createElement("strong", { className: "text-violet-900" }, "Where used:"), " ", pkg.whereUsed), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-slate-700" }, /* @__PURE__ */ import_react.default.createElement("strong", { className: "text-emerald-900" }, "You build:"), " ", pkg.whatYouBuild))
      )))), /* @__PURE__ */ import_react.default.createElement("div", { className: "rounded-xl bg-slate-900 border border-slate-800 p-4 flex flex-col justify-between" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2 mb-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-mono font-semibold text-teal-400" }, "Windows PowerShell Setup & Verification Commands"), /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          type: "button",
          onClick: () => {
            var _a;
            (_a = navigator.clipboard) == null ? void 0 : _a.writeText(SETUP_COMMANDS_TEXT);
            setCopiedSetup(true);
            setTimeout(() => setCopiedSetup(false), 2200);
          },
          className: "text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
        },
        copiedSetup ? "Copied!" : "Copy Commands"
      )), /* @__PURE__ */ import_react.default.createElement("pre", { className: "text-[11px] font-mono text-slate-100 overflow-x-auto leading-relaxed" }, /* @__PURE__ */ import_react.default.createElement("code", null, SETUP_COMMANDS_TEXT))))))), /* @__PURE__ */ import_react.default.createElement("section", { className: "max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-col md:flex-row md:items-center justify-between gap-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-2", role: "tablist", "aria-label": "Roadmap Phase Filters" }, [
        { id: "all", label: "All Phases (6)" },
        { id: "phase-1", label: "Phase 1: Foundations" },
        { id: "phase-2", label: "Phase 2: Autograd" },
        { id: "phase-3", label: "Phase 3: Tokenization" },
        { id: "phase-4", label: "Phase 4: Transformer" },
        { id: "phase-5", label: "Phase 5: Pretraining" },
        { id: "phase-6", label: "Phase 6: Fine-Tune & Serve" }
      ].map((tab) => {
        const isMatch = currentFilter === tab.id;
        return /* @__PURE__ */ import_react.default.createElement(
          "button",
          {
            key: tab.id,
            type: "button",
            onClick: () => setCurrentFilter(tab.id),
            className: `px-3 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${isMatch ? "bg-[#17394D] text-white shadow-xs" : "bg-white text-slate-600 hover:text-[#17394D] border border-slate-200/90 hover:border-slate-300"}`
          },
          tab.label
        );
      })), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2 self-start md:self-auto" }, /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          type: "button",
          onClick: () => setCollapsedPhases(/* @__PURE__ */ new Set()),
          className: "text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
        },
        "Expand All"
      ), /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          type: "button",
          onClick: () => setCollapsedPhases(new Set(ROADMAP_PHASES.map((p) => p.id))),
          className: "text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
        },
        "Collapse All"
      )))), /* @__PURE__ */ import_react.default.createElement("main", { className: "max-w-6xl mx-auto px-4 sm:px-6 py-6 pb-24" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "relative", ref: timelineRef }, /* @__PURE__ */ import_react.default.createElement("div", { className: "hidden md:block absolute left-7 top-4 bottom-8 w-1 bg-slate-200/90 rounded-full overflow-hidden" }, /* @__PURE__ */ import_react.default.createElement(
        "div",
        {
          className: "w-full bg-gradient-to-b from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-150 ease-out",
          style: { height: `${scrollSpinePct}%` }
        }
      )), /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-8" }, filteredPhases.map((phase) => {
        const theme = getAccentClasses(phase.accent);
        const phaseDoneCount = phase.topics.filter((t) => completedIds.has(t.id)).length;
        const phaseTotal = phase.topics.length;
        const pct = Math.round(phaseDoneCount / phaseTotal * 100);
        const isPhaseComplete = phaseDoneCount === phaseTotal;
        const isCollapsed = collapsedPhases.has(phase.id);
        return /* @__PURE__ */ import_react.default.createElement("article", { key: phase.id, id: phase.id, className: "relative md:pl-20" }, /* @__PURE__ */ import_react.default.createElement(
          "div",
          {
            className: `hidden md:flex absolute left-0 top-5 w-14 h-14 rounded-2xl border-2 ${isPhaseComplete ? "border-emerald-500 bg-emerald-500 text-white" : theme.nodeRing} shadow-sm items-center justify-center font-mono font-extrabold text-base z-10 transition-colors`
          },
          isPhaseComplete ? "\u2713" : phase.number
        ), /* @__PURE__ */ import_react.default.createElement("div", { className: "bg-white rounded-2xl border border-slate-200/90 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden" }, /* @__PURE__ */ import_react.default.createElement(
          "div",
          {
            onClick: () => togglePhaseCollapse(phase.id),
            className: "p-5 sm:p-6 cursor-pointer select-none flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors"
          },
          /* @__PURE__ */ import_react.default.createElement("div", { className: "space-y-1.5 flex-1" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap items-center gap-2" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "md:hidden font-mono text-xs font-bold px-2 py-0.5 rounded-md bg-[#17394D] text-white" }, "Phase ", phase.number), /* @__PURE__ */ import_react.default.createElement("span", { className: `text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full border ${theme.badgeBg}` }, phase.duration), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs font-medium text-slate-400" }, "\u2022 ", phaseDoneCount, "/", phaseTotal, " modules completed")), /* @__PURE__ */ import_react.default.createElement("h2", { className: "text-lg sm:text-xl font-extrabold text-[#17394D] tracking-tight" }, phase.title), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-xs sm:text-sm text-slate-600 leading-relaxed" }, phase.subtitle), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-2.5 p-3 rounded-xl bg-sky-50/75 border border-sky-200/80 text-xs text-[#17394D] leading-relaxed" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-sky-900 uppercase tracking-wider text-[10px] mr-1.5" }, "Plain-English Big Picture:"), phase.plainBigPicture), phase.liveAiBridge && /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-2.5 p-3.5 rounded-xl bg-gradient-to-r from-violet-50/85 via-indigo-50/60 to-white border border-violet-200/85 text-xs text-[#17394D] leading-relaxed" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-1.5 mb-1" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "w-2 h-2 rounded-full bg-violet-600 animate-pulse" }), /* @__PURE__ */ import_react.default.createElement("span", { className: "font-extrabold text-violet-950 uppercase tracking-wider text-[10px]" }, phase.liveAiBridge.headline)), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-slate-700 mb-2" }, phase.liveAiBridge.summary), /* @__PURE__ */ import_react.default.createElement("ul", { className: "space-y-1" }, phase.liveAiBridge.examples.map((ex, idx) => /* @__PURE__ */ import_react.default.createElement("li", { key: idx, className: "flex items-start gap-1.5 text-[11px] text-[#17394D]" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-violet-600 font-bold shrink-0 mt-px" }, "\u2192"), /* @__PURE__ */ import_react.default.createElement("span", null, ex))))), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-2 flex flex-wrap items-center gap-1.5 pt-0.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-1" }, "Jargon Decoder:"), phase.jargonDecoder.map((item) => /* @__PURE__ */ import_react.default.createElement(
            "span",
            {
              key: item.term,
              className: "inline-flex items-center gap-1 text-[11px] bg-slate-50 border border-slate-200/90 rounded-lg px-2 py-0.5 text-slate-700"
            },
            /* @__PURE__ */ import_react.default.createElement("strong", { className: "font-mono text-indigo-700" }, item.term),
            /* @__PURE__ */ import_react.default.createElement("span", { className: "text-slate-400" }, "="),
            /* @__PURE__ */ import_react.default.createElement("span", null, item.def)
          )))),
          /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between sm:justify-end gap-4 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "w-28" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex justify-between text-[11px] font-mono font-semibold text-slate-500 mb-1" }, /* @__PURE__ */ import_react.default.createElement("span", null, "Progress"), /* @__PURE__ */ import_react.default.createElement("span", null, pct, "%")), /* @__PURE__ */ import_react.default.createElement("div", { className: "w-full h-1.5 bg-slate-100 rounded-full overflow-hidden" }, /* @__PURE__ */ import_react.default.createElement(
            "div",
            {
              className: `h-full ${isPhaseComplete ? "bg-emerald-500" : theme.bar} rounded-full transition-all duration-500`,
              style: { width: `${pct}%` }
            }
          ))), /* @__PURE__ */ import_react.default.createElement(
            "button",
            {
              type: "button",
              "aria-label": "Toggle phase details",
              className: `w-8 h-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center transition-transform duration-300 ${isCollapsed ? "rotate-180" : ""}`
            },
            /* @__PURE__ */ import_react.default.createElement("svg", { className: "w-4 h-4", fill: "none", stroke: "currentColor", strokeWidth: "2.2", viewBox: "0 0 24 24" }, /* @__PURE__ */ import_react.default.createElement("path", { strokeLinecap: "round", strokeLinejoin: "round", d: "M19 9l-7 7-7-7" }))
          ))
        ), !isCollapsed && /* @__PURE__ */ import_react.default.createElement("div", { className: "px-5 sm:px-6 pb-6 pt-2 border-t border-slate-100 space-y-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-2 rounded-xl bg-gradient-to-r from-indigo-50/80 via-sky-50/45 to-white border border-indigo-200/80 p-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-2 mb-2.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-extrabold uppercase tracking-wider text-indigo-950" }, "What You Will Learn in Phase ", phase.number, " \u2014 Click Any Module to Open Its Interactive Lesson Page")), /* @__PURE__ */ import_react.default.createElement("div", { className: "flex flex-wrap gap-2" }, phase.topics.map((t) => /* @__PURE__ */ import_react.default.createElement(
          "button",
          {
            key: t.id,
            type: "button",
            onClick: () => setActiveTopicId(t.id),
            className: "inline-flex items-center gap-1.5 text-left bg-white/90 hover:bg-indigo-50/80 border border-indigo-200/75 hover:border-indigo-400 rounded-lg px-2.5 py-1.5 text-xs text-[#17394D] font-semibold transition-all shadow-2xs"
          },
          /* @__PURE__ */ import_react.default.createElement("span", { className: "font-mono text-[11px] font-bold text-indigo-700" }, t.codeNum),
          /* @__PURE__ */ import_react.default.createElement("span", null, t.title)
        )))), /* @__PURE__ */ import_react.default.createElement("div", { className: "grid grid-cols-1 md:grid-cols-2 gap-3 pt-1" }, phase.topics.map((topic) => {
          const checked = completedIds.has(topic.id);
          return /* @__PURE__ */ import_react.default.createElement(
            "div",
            {
              key: topic.id,
              onClick: () => setActiveTopicId(topic.id),
              className: `group relative flex items-start gap-3 p-4 rounded-xl border transition-all duration-200 cursor-pointer ${checked ? "bg-emerald-50/45 border-emerald-200/90" : `bg-slate-50/60 border-slate-200/80 ${theme.pillHover} hover:bg-white hover:shadow-xs`}`
            },
            /* @__PURE__ */ import_react.default.createElement("div", { className: "pt-0.5 shrink-0", onClick: (e) => e.stopPropagation() }, /* @__PURE__ */ import_react.default.createElement(
              "input",
              {
                type: "checkbox",
                checked,
                onChange: () => toggleTopic(topic.id),
                "aria-label": `Mark ${topic.title} completed`,
                className: "w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
              }
            )),
            /* @__PURE__ */ import_react.default.createElement("div", { className: "flex-1 min-w-0" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "flex items-center gap-1.5" }, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-white border border-slate-200/90 text-[#17394D]" }, topic.codeNum), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-mono text-slate-400" }, topic.duration)), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[11px] font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-md group-hover:bg-indigo-600 group-hover:text-white transition-all shrink-0" }, "Open Lesson Page \u2192")), /* @__PURE__ */ import_react.default.createElement(
              "h3",
              {
                className: `text-xs sm:text-sm font-bold leading-snug mt-1.5 ${checked ? "text-slate-500 line-through decoration-emerald-400" : "text-[#17394D] group-hover:text-indigo-950"}`
              },
              topic.title
            ), /* @__PURE__ */ import_react.default.createElement(
              "div",
              {
                className: `mt-2 p-2.5 rounded-lg bg-sky-50/70 border border-sky-200/70 text-[11px] leading-relaxed ${checked ? "text-slate-400" : "text-[#17394D]"}`
              },
              /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-sky-900" }, "What is this? "),
              topic.plainEnglish
            ), topic.liveExample && /* @__PURE__ */ import_react.default.createElement(
              "div",
              {
                className: `mt-2 p-2.5 rounded-lg bg-violet-50/75 border border-violet-200/80 text-[11px] leading-relaxed ${checked ? "text-slate-400" : "text-[#17394D]"}`
              },
              /* @__PURE__ */ import_react.default.createElement("span", { className: "font-bold text-violet-900" }, "Live AI In Action: "),
              topic.liveExample
            ), /* @__PURE__ */ import_react.default.createElement("ul", { className: "mt-2.5 space-y-1 border-t border-slate-200/60 pt-2" }, topic.concepts.map((concept, idx) => /* @__PURE__ */ import_react.default.createElement(
              "li",
              {
                key: idx,
                className: `flex items-start gap-1.5 text-[11px] leading-snug ${checked ? "text-slate-400" : "text-slate-600"}`
              },
              /* @__PURE__ */ import_react.default.createElement("span", { className: "text-teal-600 font-bold shrink-0 mt-px" }, "\u2022"),
              /* @__PURE__ */ import_react.default.createElement("span", null, concept)
            ))))
          );
        })), /* @__PURE__ */ import_react.default.createElement("div", { className: "rounded-xl bg-gradient-to-r from-emerald-50/80 via-teal-50/45 to-white border border-emerald-200/85 p-4" }, /* @__PURE__ */ import_react.default.createElement("div", { className: "text-[11px] font-extrabold uppercase tracking-wider text-emerald-900 mb-2.5" }, "What You Will Achieve & Master After Phase ", phase.number), /* @__PURE__ */ import_react.default.createElement("ul", { className: "grid grid-cols-1 md:grid-cols-3 gap-2.5" }, phase.achievements.map((item, idx) => /* @__PURE__ */ import_react.default.createElement(
          "li",
          {
            key: idx,
            className: "flex items-start gap-2 bg-white/85 rounded-lg p-2.5 border border-emerald-200/60 text-xs text-[#17394D] leading-snug"
          },
          /* @__PURE__ */ import_react.default.createElement("span", { className: "text-emerald-600 font-extrabold shrink-0 mt-px" }, "\u2713"),
          /* @__PURE__ */ import_react.default.createElement("span", null, item)
        )))), /* @__PURE__ */ import_react.default.createElement("div", { className: "rounded-xl bg-gradient-to-r from-amber-50/80 via-orange-50/40 to-white border border-amber-200/80 p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "text-[10px] font-bold uppercase tracking-wider text-amber-800 block" }, "Phase ", phase.number, " Milestone Build"), /* @__PURE__ */ import_react.default.createElement("span", { className: "text-xs sm:text-sm font-bold text-[#17394D]" }, phase.miniProject)), /* @__PURE__ */ import_react.default.createElement(
          "button",
          {
            type: "button",
            onClick: () => setActiveTopicId(phase.topics[phase.topics.length - 1].id),
            className: "text-xs font-semibold text-amber-900 hover:text-amber-700 underline underline-offset-2 self-start sm:self-center shrink-0"
          },
          "Open Milestone Lesson \u2192"
        )))));
      }))), /* @__PURE__ */ import_react.default.createElement("div", { className: "mt-12 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6" }, /* @__PURE__ */ import_react.default.createElement("div", null, /* @__PURE__ */ import_react.default.createElement("span", { className: "inline-block text-xs font-bold uppercase tracking-wider text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md mb-2" }, "Golden Rule for Beginners"), /* @__PURE__ */ import_react.default.createElement("h3", { className: "text-xl font-bold text-[#17394D]" }, "Don't just watch tutorials\u2014print tensor shapes at every step!"), /* @__PURE__ */ import_react.default.createElement("p", { className: "text-sm text-slate-600 mt-1.5 max-w-2xl" }, "90% of bugs when building LLMs from scratch come from mismatched matrix dimensions", " ", /* @__PURE__ */ import_react.default.createElement("code", { className: "font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded text-indigo-700" }, "(Batch, Seq_Len, Embed_Dim)"), ". Start with Module 1.1's interactive lesson page below and build every layer yourself.")), /* @__PURE__ */ import_react.default.createElement("div", { className: "shrink-0 flex flex-col sm:flex-row gap-3 w-full md:w-auto" }, /* @__PURE__ */ import_react.default.createElement(
        "button",
        {
          type: "button",
          onClick: () => setActiveTopicId(ALL_TOPICS[0].id),
          className: "px-5 py-3 rounded-xl bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 text-white font-semibold text-sm shadow-sm transition-all text-center"
        },
        "Open Module 1.1 Lesson Page \u2192"
      )))))
    ), toast && /* @__PURE__ */ import_react.default.createElement("div", { className: "fixed bottom-5 right-5 z-50 bg-[#17394D] text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3 text-xs sm:text-sm" }, /* @__PURE__ */ import_react.default.createElement("span", null, toast.message), toast.confirm && /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => {
          setCompletedIds(/* @__PURE__ */ new Set());
          setToast({ message: "Roadmap progress cleared.", confirm: false });
          setTimeout(() => setToast(null), 2500);
        },
        className: "px-2.5 py-1 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-semibold text-xs"
      },
      "Yes, Reset"
    ), /* @__PURE__ */ import_react.default.createElement(
      "button",
      {
        type: "button",
        onClick: () => setToast(null),
        className: "text-slate-300 hover:text-white font-semibold text-xs"
      },
      "Dismiss"
    )));
  }

export default App;