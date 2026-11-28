import React, { useState, useEffect, useMemo, useRef } from "react";

// =========================================================================
// 1. CURRICULUM & SETUP DATA DEFINITIONS
// =========================================================================
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

const PIPELINE_STAGES = [
  {
    id: 1,
    name: "1. Tokenizer",
    sub: "Text -> Token IDs",
    phaseId: "phase-3",
    badge: "Phase 3 • Tokenization",
    desc: 'Raw text is split into subword integer IDs using Byte-Pair Encoding: "The capital of France is" -> [464, 3139, 286, 4881, 318].'
  },
  {
    id: 2,
    name: "2. Embeddings",
    sub: "IDs -> Dense Vectors",
    phaseId: "phase-3",
    badge: "Phase 3 • Vector Space",
    desc: "Each token ID looks up a learned vector (e.g., 768 floats) plus positional info (RoPE/Positional Embedding) so order matters."
  },
  {
    id: 3,
    name: "3. Self-Attention",
    sub: "Context Mixing (Q·Kᵀ)",
    phaseId: "phase-4",
    badge: "Phase 4 • Heart of GPT",
    desc: "Tokens communicate via Queries, Keys, and Values with a causal mask so 'France' attends back to 'capital' without peeking at future words."
  },
  {
    id: 4,
    name: "4. Feed-Forward",
    sub: "MLP Knowledge Store",
    phaseId: "phase-4",
    badge: "Phase 4 • Transformer Block",
    desc: "Each token passes through LayerNorm/RMSNorm, a 2-layer MLP with GELU/SwiGLU activation, and residual skip connections."
  },
  {
    id: 5,
    name: "5. LM Head + Softmax",
    sub: "Logits -> Probabilities",
    phaseId: "phase-5",
    badge: "Phase 5 • Next-Token Prediction",
    desc: "Final hidden vectors project back to vocabulary size (e.g., 50,257 logits). Softmax turns them into probabilities—'Paris' scores highest!"
  },
  {
    id: 6,
    name: "6. SFT & Local Run",
    sub: "Assistant Alignment",
    phaseId: "phase-6",
    badge: "Phase 6 • Post-Training",
    desc: "Fine-tune the base predictor on Q&A instructions (SFT/LoRA), quantize weights, and serve locally via Ollama or FastAPI."
  }
];

const SETUP_SOFTWARES = [
  {
    name: "Python 3.11 (64-bit)",
    url: "https://www.python.org/downloads/",
    role: "Core language runtime for writing models from scratch, training loops, and data pipelines.",
    whereUsed: "Used across every AI lab (OpenAI, Meta, Hugging Face) as the control plane orchestrating C++/CUDA kernels.",
    whatYouBuild: "Your isolated .venv environment, BPE tokenizer, GPT model classes, training loop, and API server.",
    winget: "winget install -e --id Python.Python.3.11"
  },
  {
    name: "Visual Studio Code",
    url: "https://code.visualstudio.com/",
    role: "Code editor, integrated terminal, Jupyter notebook runner, and tensor shape debugger.",
    whereUsed: "Daily IDE for ML engineers to inspect tensor shapes, debug training runs, and browse repositories.",
    whatYouBuild: "Modular Python scripts (tokenizer.py, model.py, train.py) and interactive notebooks.",
    winget: "winget install -e --id Microsoft.VisualStudioCode"
  },
  {
    name: "Git for Windows",
    url: "https://git-scm.com/downloads/win",
    role: "Version control for tracking model experiments, checkpoint scripts, and pushing to GitHub.",
    whereUsed: "Every collaborative AI codebase and open-source model repository on GitHub/Hugging Face.",
    whatYouBuild: "Clean commit history tracking every phase from micrograd to your pretrained mini-GPT.",
    winget: "winget install -e --id Git.Git"
  },
  {
    name: "Ollama",
    url: "https://ollama.com/download",
    role: "Runs quantized open-source LLMs (Llama 3, Qwen, Gemma) locally on your CPU or GPU.",
    whereUsed: "Local developer workstations and private on-premise enterprise AI deployments.",
    whatYouBuild: "A local reference baseline to compare latency and answer quality against your custom GPT.",
    winget: "winget install -e --id Ollama.Ollama"
  }
];

const SETUP_PACKAGES = [
  { name: "torch", desc: "PyTorch autograd & GPU tensor engine", whereUsed: "Meta, OpenAI & Tesla AI training clusters", whatYouBuild: "Self-attention, Transformer blocks & GPTModel" },
  { name: "numpy", desc: "Fast C-backed matrix & vector math", whereUsed: "Every scientific & ML preprocessing pipeline", whatYouBuild: "Matrix multiplication & Softmax visualizers" },
  { name: "transformers", desc: "Reference architectures & weight loaders", whereUsed: "Hugging Face ecosystem & fine-tuning pipelines", whatYouBuild: "Weight loader mapping GPT-2 into your code" },
  { name: "tokenizers", desc: "Rust-accelerated subword BPE tokenizers", whereUsed: "Production LLM inference engines (vLLM, TGI)", whatYouBuild: "Benchmark comparison for your pure-Python BPE" },
  { name: "datasets", desc: "Streaming text corpora & dataset loaders", whereUsed: "Pretraining & SFT data curation pipelines", whatYouBuild: "TinyStories & instruction Q&A data loaders" },
  { name: "jupyterlab", desc: "Interactive notebooks for step-by-step math", whereUsed: "Research prototyping & activation diagnostics", whatYouBuild: "Interactive attention heatmap & loss notebooks" },
  { name: "pandas", desc: "Tabular dataset cleaning & inspection", whereUsed: "Filtering & deduplicating SFT instruction data", whatYouBuild: "Clean Q&A instruction formatting pipelines" },
  { name: "matplotlib", desc: "Plotting loss curves & attention heatmaps", whereUsed: "Training run diagnostics & paper figures", whatYouBuild: "Train vs. validation loss & cosine LR plots" },
  { name: "pytest", desc: "Automated shape & gradient unit testing", whereUsed: "CI/CD checks in production ML repositories", whatYouBuild: "Automated tensor-shape & prompt quality tests" },
  { name: "mlflow", desc: "Experiment tracking for hyperparams & loss", whereUsed: "Enterprise ML teams comparing training runs", whatYouBuild: "Local experiment dashboard for LR & loss runs" },
  { name: "fastapi", desc: "Typed Python REST API framework", whereUsed: "Production model-serving microservices", whatYouBuild: "/v1/generate HTTP endpoint for your GPT" },
  { name: "uvicorn", desc: "High-speed ASGI web server for FastAPI", whereUsed: "Hosting Python AI APIs in containers", whatYouBuild: "Localhost server running your custom LLM API" },
  { name: "requests", desc: "HTTP client for downloading corpora & testing APIs", whereUsed: "Fetching datasets & querying inference endpoints", whatYouBuild: "Client script testing your FastAPI & Ollama APIs" },
  { name: "lxml", desc: "Fast XML/HTML text extraction & parsing", whereUsed: "Scraping & cleaning raw web text for pretraining", whatYouBuild: "Clean corpus extractor stripping HTML tags" }
];

const SETUP_COMMANDS_TEXT = `mkdir VIT-Automation
cd VIT-Automation
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib jupyterlab transformers datasets tokenizers pytest lxml requests mlflow fastapi uvicorn
python -m pip install torch
python -c "import torch, transformers, datasets, mlflow, fastapi; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"`;

const ROADMAP_PHASES = [
  {
    id: "phase-1",
    number: "01",
    category: "foundations",
    accent: "teal",
    duration: "Week 1–2",
    title: "Python, NumPy & PyTorch Foundations for LLMs",
    subtitle: "Master clean Python OOP, NumPy broadcasting, the (B, T, C) tensor shape mental model, and probability/loss math before writing neural nets.",
    plainBigPicture: "An LLM is really just a giant calculator that reads numbers representing words and predicts the probability of the next word. Before building the brain, you need Python (the glue language), NumPy (a fast math library for grids of numbers), and PyTorch (an AI framework that runs those grids on your GPU).",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 1 Skills",
      summary: "Every production LLM API (OpenAI, Anthropic, Ollama) relies directly on these exact primitives on every single request:",
      examples: [
        "OpenAI's 'logprobs' API parameter exposes the exact raw Logits -> Softmax probability percentages you build in Module 1.5.",
        "Production inference servers batch multiple users' prompts together into a single (B, T, C) 3D tensor (Module 1.3) to maximize GPU throughput."
      ]
    },
    jargonDecoder: [
      { term: "LLM", def: "Next-word probability predictor" },
      { term: "NumPy", def: "Fast grid-of-numbers math library" },
      { term: "PyTorch", def: "AI framework with GPU acceleration" },
      { term: "Tensor", def: "Multi-dimensional spreadsheet of numbers" },
      { term: "Logits", def: "Raw unscaled word scores" },
      { term: "Softmax", def: "Converts raw scores into % probabilities" }
    ],
    achievements: [
      "Confidently write clean, modular Python classes and @dataclass configs inside an isolated .venv environment.",
      "Trace (Batch, Sequence, Embedding) 3D tensor shapes across matrix multiplications without shape-mismatch errors.",
      "Convert raw model logits into calibrated token probabilities using Softmax and compute Cross-Entropy loss."
    ],
    miniProject: "Build a Matrix Multiplication & Softmax Temperature Visualizer in NumPy/PyTorch",
    topics: [
      {
        id: "p1-t1",
        codeNum: "1.1",
        animType: "python_oop",
        title: "Python Data Structures, OOP & Clean ML Code",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase1_foundations",
          fileName: "01_data_structures_oop_clean_ml.py",
          steps: [
            "Open your VIT-Automation folder in VS Code (File -> Open Folder).",
            "Open the VS Code integrated terminal (Ctrl + `) and confirm (.venv) is active in your prompt.",
            "Create a subfolder named phase1_foundations and a new file inside it called 01_data_structures_oop_clean_ml.py.",
            "Paste the complete Python starter code below, save (Ctrl + S), and run it with python phase1_foundations/01_data_structures_oop_clean_ml.py."
          ],
          starterCode: `from dataclasses import dataclass

# 1. CLEAN ML CODE: Frozen @dataclass config with invariant validation
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6

    def __post_init__(self) -> None:
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.emb_dim % self.n_heads == 0, "emb_dim must divide evenly by n_heads!"

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

# 2. PYTHON DATA STRUCTURES + OOP: Encapsulating list, tuple, dict, and set
class SimpleVocab:
    def __init__(self, words: list[str]) -> None:
        # set: O(1) uniqueness tracking while preserving order in a list
        unique_words: list[str] = []
        seen: set[str] = set()
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        # dict: O(1) key -> value lookup maps
        self._word_to_id: dict[str, int] = {w: idx for idx, w in enumerate(unique_words)}
        self._id_to_word: dict[int, str] = {idx: w for idx, w in enumerate(unique_words)}

    def __len__(self) -> int:
        return len(self._word_to_id)

    def __repr__(self) -> str:
        return f"SimpleVocab(size={len(self)})"

    def encode(self, sentence: str) -> list[int]:
        return [self._word_to_id[w] for w in sentence.split() if w in self._word_to_id]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_word[i] for i in ids)

    def expected_tensor_shape(self, batch_size: int, seq_len: int, cfg: GPTConfig) -> tuple[int, int, int]:
        # tuple: immutable fixed dimensions (Batch, Tokens, Channels)
        return (batch_size, seq_len, cfg.emb_dim)

if __name__ == "__main__":
    cfg = GPTConfig()
    print(f"Config loaded -> width={cfg.emb_dim}, head_dim={cfg.head_dim}")

    vocab = SimpleVocab(["the", "capital", "of", "france", "is", "paris", "the"])
    encoded = vocab.encode("the capital of france")
    print(f"{vocab} | Encoded (list): {encoded} | Decoded: '{vocab.decode(encoded)}'")
    print(f"Expected 3D shape (tuple): {vocab.expected_tensor_shape(2, len(encoded), cfg)}")`,
          runCmd: "python phase1_foundations/01_data_structures_oop_clean_ml.py",
          expectedOutput: `Config loaded -> width=384, head_dim=64
SimpleVocab(size=6) | Encoded (list): [0, 1, 2, 3] | Decoded: 'the capital of france'
Expected 3D shape (tuple): (2, 4, 384)`
        },
        plainEnglish: "Python's four core data structures (list, tuple, dict, set) organize your tokens, shapes, and vocabulary mappings; Object-Oriented Programming (OOP) bundles data and methods into reusable classes; and clean ML code uses typed configs, shape assertions, and modular functions so models are easy to debug.",
        liveExample: "PyTorch's nn.Module and Hugging Face's PreTrainedConfig rely directly on OOP inheritance, dictionaries (state_dict & vocab), tuples (tensor shapes), and clean modular methods.",
        jargonTerms: [
          { term: "list & tuple", def: "list = ordered mutable sequence ([10, 20]); tuple = ordered immutable sequence ((B, T, C) tensor shapes)." },
          { term: "dict & set", def: "dict = fast key-value lookup map ({'paris': 5}); set = unordered collection of unique items (visited graph nodes)." },
          { term: "OOP (Class, Encapsulation, Inheritance, Polymorphism)", def: "Structuring code into reusable blueprints that bundle state & methods, extend base classes via super(), and share standard interfaces." },
          { term: "Clean ML Code", def: "Separating configs (@dataclass), data pipelines, model classes, and training loops with type hints and shape assertions." }
        ],
        why: "Building an LLM from scratch combines tokenizers, datasets, attention heads, Transformer blocks, and trainers—mastering Python data structures, OOP, and clean ML structure prevents tangled, hard-to-debug scripts.",
        lessonSections: [
          {
            heading: "1. Python Data Structures: What They Are & What Each Contains",
            body: "Python provides four built-in containers used across every ML pipeline: Lists (ordered, mutable sequences like token IDs [464, 3139]), Tuples (ordered, immutable sequences like fixed tensor shapes (2, 8, 64)), Dictionaries (key-value hash maps like token-to-ID vocabularies, BPE merge rules, and PyTorch state_dict checkpoints), and Sets (unordered unique elements for deduplicating tokens or tracking visited nodes in autograd).",
            codeExample: `token_ids: list[int] = [464, 3139, 286]             # Ordered, mutable list
tensor_shape: tuple[int, int, int] = (2, 8, 64)     # Immutable 3D shape
vocab: dict[str, int] = {"<pad>": 0, "the": 1}      # Key -> Value lookup
visited_nodes: set[str] = {"embed", "attn_0"}       # Unique items only`
          },
          {
            heading: "2. OOP Concepts in Python for Machine Learning",
            body: "Object-Oriented Programming organizes code around classes and objects: Encapsulation keeps internal attributes (like vocabulary maps or layer weights) safe behind methods; Inheritance lets a subclass extend a parent class (like subclassing torch.nn.Module and calling super().__init__()); Polymorphism lets interchangeable components share methods (encode/decode or forward); and Abstraction hides internal complexity behind clean method calls.",
            codeExample: `class BaseTokenizer:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

class WordTokenizer(BaseTokenizer):
    def __init__(self, mapping: dict[str, int]) -> None:
        super().__init__()
        self._mapping = mapping

    def encode(self, text: str) -> list[int]:
        return [self._mapping[w] for w in text.split() if w in self._mapping]`
          },
          {
            heading: "3. What Clean ML Code Looks Like",
            body: "Clean ML code separates hyperparameters (@dataclass configs), data loading, model architecture, and training loops into small, testable units; uses type annotations (list[int], dict[str, int]); checks invariants early with assert (such as emb_dim % n_heads == 0); and keeps experiments reproducible inside an isolated .venv environment.",
            codeExample: `from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
    emb_dim: int = 384
    n_heads: int = 6

    def __post_init__(self) -> None:
        assert self.emb_dim % self.n_heads == 0, "emb_dim must divide evenly by n_heads"`
          }
        ],
        concepts: [
          "Python data structures: what lists, tuples, dicts, and sets are, what each contains, and when to use each in ML.",
          "Python OOP concepts: classes, instances, encapsulation, inheritance (super().__init__()), polymorphism, abstraction, and dunder methods (__len__, __repr__, __getitem__).",
          "Clean ML code practices: @dataclass configs, type hints, shape/invariant assertions, isolated .venv environments, and separating data, model, and training code."
        ],
        code: `from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6

    def __post_init__(self) -> None:
        assert self.emb_dim % self.n_heads == 0

cfg = GPTConfig()
print(f"Model width: {cfg.emb_dim}, Head dim: {cfg.emb_dim // cfg.n_heads}")`,
        quiz: {
          question: "Which Python data structure best represents a fixed tensor shape like (batch_size, seq_len, emb_dim), and which best maps token strings to integer IDs?",
          options: [
            "A list for tensor shapes and a set for token-to-ID mappings",
            "A tuple for fixed tensor shapes and a dict for token-to-ID mappings",
            "A dict for tensor shapes and a tuple for token-to-ID mappings"
          ],
          answerIndex: 1,
          explanation: "Tuples store ordered, immutable dimensions like shapes cleanly, while dicts provide fast key-value lookups from token strings to integer IDs."
        },
        pitfall: "Putting data preparation, model layers, training loops, and file paths into one giant script without config classes, type hints, or shape assertions—and installing packages globally instead of inside .venv.",
        project: "Write a clean Python module containing a @dataclass GPTConfig with assertions, a Vocabulary OOP class using dict/set/list/tuple, and __len__/__repr__/encode/decode methods.",
        resourceTitle: "Sebastian Raschka — Build an LLM From Scratch (Environment & Python Setup)",
        resourceNote: "Start here to structure clean Python classes and virtual environments before writing PyTorch models.",
        links: [
          { type: "YouTube", label: "Sebastian Raschka — Build an LLM from Scratch 1: Set Up Your Code Environment", url: "https://www.youtube.com/watch?v=yAcWnfsZhzo" },
          { type: "Community Guide", label: "Real Python — Object-Oriented Programming & Data Classes Guide", url: "https://realpython.com/python-data-classes/" }
        ]
      },
      {
        id: "p1-t2",
        codeNum: "1.2",
        animType: "broadcasting",
        title: "Vectors, Matrices & Broadcasting in NumPy",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase1_foundations",
          fileName: "02_numpy_vectors_broadcasting.py",
          steps: [
            "In VS Code inside phase1_foundations/, create a new file named 02_numpy_vectors_broadcasting.py.",
            "Paste the NumPy vector similarity & broadcasting code below and save (Ctrl + S).",
            "Run the file in your (.venv) terminal and inspect the broadcasted shapes and dot-product similarity grid."
          ],
          starterCode: `import numpy as np

# 1. Word vectors (4 tokens, each represented by 3 semantic features)
tokens = ["king", "queen", "apple", "banana"]
X = np.array([
    [0.92, 0.85, 0.10],  # king
    [0.89, 0.88, 0.12],  # queen
    [0.10, 0.15, 0.95],  # apple
    [0.12, 0.11, 0.91],  # banana
], dtype=np.float32)

# 2. Broadcasting: add a 1D bias vector (3,) across all 4 rows (4, 3)
bias = np.array([0.05, -0.05, 0.02], dtype=np.float32)
X_shifted = X + bias
print("Original shape:", X.shape, "| After broadcasting bias:", X_shifted.shape)

# 3. Pairwise dot-product similarity matrix (4x3) @ (3x4) -> (4x4)
similarity = np.round(X @ X.T, 2)
print("\\nPairwise Similarity Matrix (king/queen vs apple/banana):")
print(similarity)`,
          runCmd: "python phase1_foundations/02_numpy_vectors_broadcasting.py",
          expectedOutput: `Original shape: (4, 3) | After broadcasting bias: (4, 3)

Pairwise Similarity Matrix (king/queen vs apple/banana):
[[1.58 1.58 0.31 0.29]
 [1.58 1.58 0.33 0.31]
 [0.31 0.33 0.94 0.89]
 [0.29 0.31 0.89 0.85]]`
        },
        plainEnglish: "Standard Python lists are far too slow for AI because Python checks data types one item at a time. NumPy stores numbers in contiguous C-memory blocks so your CPU can multiply millions of numbers simultaneously.",
        liveExample: "Vector search engines (Pinecone, Chroma, pgvector) powering RAG chatbots find relevant documents using the exact dot-product cosine similarity math here.",
        jargonTerms: [
          { term: "NumPy", def: "Numerical Python—a C-powered library for ultra-fast array math." },
          { term: "Vector", def: "A 1D list of numbers representing a word's meaning." },
          { term: "Matrix", def: "A 2D grid (rows x columns) of numbers." },
          { term: "Broadcasting", def: "Automatically stretching a smaller row/column to add onto a bigger table." }
        ],
        why: "Token embeddings are vectors, weight layers are matrices, and NumPy broadcasting rules govern how biases and attention masks combine automatically.",
        lessonSections: [
          {
            heading: "1. Dot Products Measure Semantic Similarity",
            body: "When two word vectors point in a similar direction, their dot product (A @ B) is large and positive. Self-attention uses this exact operation to score how strongly two words relate.",
            codeExample: "import numpy as np\nking = np.array([0.9, 0.8, 0.1])\nqueen = np.array([0.88, 0.85, 0.12])\nprint('Similarity score:', np.dot(king, queen))"
          }
        ],
        concepts: [
          "Scalars (0D), Vectors (1D), Matrices (2D), and Tensors (3D/4D) as nested numeric arrays.",
          "Element-wise operations vs. dot products (np.dot / @) as geometric cosine similarity.",
          "Broadcasting rules: aligning trailing dimensions (e.g., adding a (1, C) bias vector to a (T, C) token matrix)."
        ],
        code: `import numpy as np\n\nX = np.array([[1.0, 0.2, -0.5],\n              [0.9, 0.1, -0.4],\n              [-0.8, 0.7, 0.3],\n              [0.0, -0.5, 0.9]])\n\nbias = np.array([0.1, -0.1, 0.05])\nprint("Broadcasted sum shape:", (X + bias).shape)\nprint("Pairwise similarity (4x4):\\n", np.round(X @ X.T, 2))`,
        quiz: {
          question: "If X has shape (4, 3) and bias has shape (3,), what happens when you compute X + bias?",
          options: [
            "NumPy raises a ValueError because shapes don't match",
            "NumPy broadcasts bias across all 4 rows without copying memory, producing shape (4, 3)",
            "NumPy truncates X to shape (3,)"
          ],
          answerIndex: 1,
          explanation: "Trailing dimensions match (3 == 3), so NumPy virtually stretches the 1D bias vector across all 4 rows."
        },
        pitfall: "Confusing element-wise multiplication (*) with matrix multiplication (@). In Python, A * B multiplies matching cells; A @ B performs row-by-column dot products.",
        project: "Implement vector cosine similarity in NumPy and test how similar word vectors have dot products near +1.",
        resourceTitle: "3Blue1Brown — Essence of Linear Algebra (Chapters 1 & 4)",
        resourceNote: "Visualizes vectors, linear transformations, matrix multiplication, and dot products geometrically.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Vectors | Essence of Linear Algebra Ch. 1", url: "https://www.youtube.com/watch?v=fNk_zzaMoSs" },
          { type: "Community Guide", label: "Jay Alammar — A Visual Intro to NumPy and Data Representation", url: "https://jalammar.github.io/visual-numpy/" }
        ]
      },
      {
        id: "p1-t3",
        codeNum: "1.3",
        animType: "tensor_btc",
        title: "Matrix Multiplication & Tensor Shapes (B, T, C) Mental Model",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase1_foundations",
          fileName: "03_tensor_shapes_btc.py",
          steps: [
            "Create 03_tensor_shapes_btc.py inside your phase1_foundations/ folder in VS Code.",
            "Paste the PyTorch 3D shape-tracing script below to see how (B, T, C) splits into attention heads and merges back.",
            "Run it in your (.venv) terminal and verify every printed tensor shape."
          ],
          starterCode: `import torch
import torch.nn as nn

B, T, C = 2, 5, 64   # 2 sentences, 5 tokens each, 64 channels per token
n_heads = 4
head_dim = C // n_heads

# 1. Simulate token embeddings of shape (B, T, C)
x = torch.randn(B, T, C)
print("1. Input embeddings (B, T, C):", x.shape)

# 2. Linear projection preserves B and T, transforms C -> 128
proj = nn.Linear(C, 128)
print("2. After Linear(64, 128):      ", proj(x).shape)

# 3. Split (B, T, C) into 4 parallel heads -> (B, n_heads, T, head_dim)
multi_head = x.view(B, T, n_heads, head_dim).transpose(1, 2)
print("3. Split into 4 heads:         ", multi_head.shape)

# 4. Merge heads safely back using .contiguous().view(B, T, C)
merged = multi_head.transpose(1, 2).contiguous().view(B, T, C)
print("4. Merged back to (B, T, C):   ", merged.shape)`,
          runCmd: "python phase1_foundations/03_tensor_shapes_btc.py",
          expectedOutput: `1. Input embeddings (B, T, C): torch.Size([2, 5, 64])
2. After Linear(64, 128):       torch.Size([2, 5, 128])
3. Split into 4 heads:          torch.Size([2, 4, 5, 16])
4. Merged back to (B, T, C):    torch.Size([2, 5, 64])`
        },
        plainEnglish: "Think of a 'Tensor' as a stack of spreadsheets. Throughout an entire LLM, data travels in a 3D box shaped (B, T, C): B is Batch size, T is Tokens in each sentence, and C is Channels (numbers describing each token).",
        liveExample: "In Llama-3 8B, every prompt flows through 32 Transformer layers with shape (B, T, 4096) before projecting to 128,256 vocabulary logits.",
        jargonTerms: [
          { term: "Tensor", def: "A generalized container of numbers (1D list, 2D table, or 3D+ cube)." },
          { term: "B (Batch Size)", def: "Number of separate sentences processed in parallel for speed." },
          { term: "T (Sequence Length)", def: "Number of tokens (word-pieces) inside a single sentence." },
          { term: "C (Embedding Dim)", def: "Size of the number list describing each individual token." }
        ],
        why: "Every single Transformer layer passes around a 3D tensor of shape (Batch, Time/Sequence, Channels/Embedding). Knowing this mental model prevents 90% of bugs.",
        lessonSections: [
          {
            heading: "1. Tracking (B, T, C) Through Linear Layers",
            body: "PyTorch's nn.Linear(C, D) always operates on the LAST axis (-1), leaving B and T completely untouched.",
            codeExample: "import torch, torch.nn as nn\nx = torch.randn(2, 5, 64)      # (B=2, T=5, C=64)\nproj = nn.Linear(64, 128)\nprint(proj(x).shape)           # -> torch.Size([2, 5, 128])"
          }
        ],
        concepts: [
          "What B (batch size), T (sequence length / context window), and C (embedding dimension) represent physically.",
          "Batched matrix multiplication: (B, T, C) @ (C, D) -> (B, T, D) transforms every token independently in parallel.",
          "Reshaping vs. transposing: why .view()/.reshape() and .transpose(-2, -1) behave differently in memory."
        ],
        code: `import torch\n\nB, T, C = 2, 5, 64\nW_out = 128\nx = torch.randn(B, T, C)\nW = torch.randn(C, W_out)\ny = x @ W\nprint("Transformed tensor shape:", y.shape)`,
        quiz: {
          question: "If tensor x has shape (4, 16, 128) and weight matrix W has shape (128, 256), what is the shape of x @ W?",
          options: ["(4, 16, 256)", "(4, 256, 16)", "(16, 128, 256)"],
          answerIndex: 0,
          explanation: "Batched matrix multiplication contracts the inner dimension 128 and preserves the batch (4) and sequence (16) dimensions."
        },
        pitfall: "Never use .view() to swap axes like (B, T, C) to (B, C, T)—that scrambles data! Always use .transpose(1, 2) when swapping dimensions.",
        project: "Write a shape-tracing script that passes a dummy (4, 16, 128) tensor through projections, splits it into 4 heads (4, 4, 16, 32), and merges it back.",
        resourceTitle: "Jay Alammar — The Illustrated Transformer (Tensor Dimension Diagrams)",
        resourceNote: "Study how input matrices keep their sequence length T constant from bottom to top of the stack.",
        links: [
          { type: "Community Guide", label: "Jay Alammar — The Illustrated Transformer (Must-Read Visual Blog)", url: "https://jalammar.github.io/illustrated-transformer/" },
          { type: "YouTube", label: "3Blue1Brown — Transformers, the tech behind LLMs (Ch. 5)", url: "https://www.youtube.com/watch?v=wjZofJX0v4M" }
        ]
      },
      {
        id: "p1-t4",
        codeNum: "1.4",
        animType: "tensor_btc",
        title: "PyTorch Tensors, Device Management & Vectorization",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase1_foundations",
          fileName: "04_device_and_vectorization.py",
          steps: [
            "Create 04_device_and_vectorization.py inside phase1_foundations/ in VS Code.",
            "Paste the device-agnostic benchmark script comparing a slow Python loop vs. vectorized PyTorch math.",
            "Run it in your (.venv) terminal to see your active hardware device and speedup."
          ],
          starterCode: `import time
import torch

# 1. Pick CUDA GPU automatically if present, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Active PyTorch device:", device)

# 2. Vectorized row-wise mean & variance (keepdim=True is critical!)
x = torch.randn(32, 128, device=device, dtype=torch.float32)
mean = x.mean(dim=-1, keepdim=True)
normalized = (x - mean) / (x.std(dim=-1, keepdim=True) + 1e-5)
print("Normalized shape:", normalized.shape, "| Row 0 mean:", round(normalized[0].mean().item(), 5))

# 3. Speed comparison: Python loop vs Vectorized dot product
a, b = torch.randn(500_000), torch.randn(500_000)
t0 = time.perf_counter()
_ = sum(float(a[i]) * float(b[i]) for i in range(50_000))
t_loop = (time.perf_counter() - t0) * 10

t1 = time.perf_counter()
_ = torch.dot(a, b)
t_vec = time.perf_counter() - t1
print(f"Estimated Python loop: {t_loop:.4f}s vs Vectorized PyTorch: {t_vec:.6f}s")`,
          runCmd: "python phase1_foundations/04_device_and_vectorization.py",
          expectedOutput: `Active PyTorch device: cpu
Normalized shape: torch.Size([32, 128]) | Row 0 mean: -0.0
Estimated Python loop: 0.3410s vs Vectorized PyTorch: 0.000185s`
        },
        plainEnglish: "PyTorch is like NumPy with two superpowers: it runs calculations on a GPU 50x faster than a CPU, and it automatically tracks gradients for learning.",
        liveExample: "Modern LLM training clusters run PyTorch in bfloat16 precision across thousands of NVIDIA H100 GPUs using identical .to(device) tensor APIs.",
        jargonTerms: [
          { term: "PyTorch", def: "Meta's open-source Python library for building and training neural networks." },
          { term: "CUDA / GPU", def: "Graphics card hardware designed to do thousands of matrix multiplications at once." },
          { term: "Vectorization", def: "Replacing slow one-by-one Python loops with single-step array math." }
        ],
        why: "PyTorch replaces slow Python for-loops with vectorized C++/CUDA tensor kernels and lets the exact same code run on CPU, NVIDIA CUDA, or Apple MPS.",
        lessonSections: [
          {
            heading: "1. Device-Agnostic Tensor Execution",
            body: "Always move both your model and your input batches to the same device variable so your code runs seamlessly on laptops and cloud GPUs.",
            codeExample: "device = 'cuda' if torch.cuda.is_available() else 'cpu'\nx = torch.randn(32, 128, device=device)"
          }
        ],
        concepts: [
          "Creating tensors, inspecting .shape, .dtype (float32, bfloat16, int64), and .device.",
          "Device-agnostic code pattern: device = 'cuda' if torch.cuda.is_available() else 'cpu'.",
          "Vectorization: replacing nested Python loops with fast tensor reductions (.sum(dim=-1), .mean(dim=-1, keepdim=True))."
        ],
        code: `import torch\n\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\nx = torch.randn(32, 128, device=device, dtype=torch.float32)\nmean = x.mean(dim=-1, keepdim=True)\nvar = x.var(dim=-1, keepdim=True, unbiased=False)\nprint(f"Running on {device} | Mean shape: {mean.shape}")`,
        quiz: {
          question: "Why do we pass keepdim=True when computing x.mean(dim=-1, keepdim=True) on a (32, 128) tensor?",
          options: [
            "To keep the output shape as (32, 1) so it broadcasts cleanly back against (32, 128)",
            "To prevent PyTorch from deleting the input tensor",
            "To force CPU execution"
          ],
          answerIndex: 0,
          explanation: "Without keepdim=True, (32, 128).mean(dim=-1) collapses to (32,), which broadcasts along the wrong dimension if subtracted from (32, 128)."
        },
        pitfall: "Forgetting keepdim=True during reductions! Without keepdim=True, (32, 128).mean(dim=-1) becomes (32,) instead of (32, 1), causing silent wrong broadcasting.",
        project: "Benchmark a pure-Python nested loop dot-product against vectorized PyTorch tensor operations on 1,000x1,000 matrices.",
        resourceTitle: "PyTorch Official Tutorials — Introduction to Tensors",
        resourceNote: "Covers tensor creation, data types, GPU device transfers, and essential tensor ops.",
        links: [
          { type: "YouTube", label: "PyTorch Official — Introduction to PyTorch Tensors", url: "https://www.youtube.com/watch?v=r7QDUPb2dCM" },
          { type: "Blog / Docs", label: "PyTorch Official Tutorial — Introduction to Tensors", url: "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html" }
        ]
      },
      {
        id: "p1-t5",
        codeNum: "1.5",
        animType: "softmax_logits",
        title: "Probability Distributions, Logits, Softmax & Cross-Entropy Loss",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase1_foundations",
          fileName: "05_logits_softmax_loss.py",
          steps: [
            "Create 05_logits_softmax_loss.py inside phase1_foundations/ in VS Code.",
            "Paste the Softmax & Cross-Entropy loss experiment below.",
            "Run it to observe how raw logits turn into probabilities and why untrained GPT loss starts at ln(50257) = 10.82."
          ],
          starterCode: `import math
import torch
import torch.nn.functional as F

# 1. Raw unnormalized scores (logits) for 4 candidate next words: ["Paris", "Lyon", "Rome", "banana"]
words = ["Paris", "Lyon", "Rome", "banana"]
logits = torch.tensor([[3.2, 1.1, 0.4, -1.5]])
target_id = torch.tensor([0])  # Index 0 ("Paris") is the true next token

probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(logits, target_id)

for w, p in zip(words, probs[0].tolist()):
    print(f"Token '{w:6s}' -> Probability: {p*100:5.1f}%")
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# 2. Sanity check: initial random loss for GPT-2 vocab size (50,257)
print(f"Expected initial GPT-2 random loss: ln(50257) = {math.log(50257):.4f}")`,
          runCmd: "python phase1_foundations/05_logits_softmax_loss.py",
          expectedOutput: `Token 'Paris ' -> Probability:  83.3%
Token 'Lyon  ' -> Probability:  10.2%
Token 'Rome  ' -> Probability:   5.1%
Token 'banana' -> Probability:   0.8%
Cross-Entropy Loss: 0.1823
Expected initial GPT-2 random loss: ln(50257) = 10.8249`
        },
        plainEnglish: "When an LLM guesses the next word, it outputs a raw score ('Logit') for every word in its dictionary. 'Softmax' turns those scores into percentages adding up to 100%, and 'Cross-Entropy Loss' grades how wrong the guess was.",
        liveExample: "Every token generated by ChatGPT or Claude comes from sampling this Softmax probability distribution over 100,000+ vocabulary logits.",
        jargonTerms: [
          { term: "Logits", def: "Raw positive or negative scores before converting to percentages." },
          { term: "Softmax", def: "Math formula turning any list of numbers into probabilities between 0% and 100%." },
          { term: "Cross-Entropy Loss", def: "Penalty score: 0 means 100% confident & right; high means confident & wrong." }
        ],
        why: "An LLM outputs unnormalized raw scores (logits) across 50,000+ vocabulary words. Softmax turns them into probabilities, and Cross-Entropy grades how confident the model was in the true next token.",
        lessonSections: [
          {
            heading: "1. Initial Loss Sanity Check: ln(vocab_size)",
            body: "Before training begins, a well-initialized model assigns roughly equal probability 1/V to every word. Thus initial Cross-Entropy loss should equal -ln(1/V) = ln(V) (around 10.82 for V=50,257).",
            codeExample: "import math\nprint('Expected initial loss for GPT-2:', math.log(50257))"
          }
        ],
        concepts: [
          "Logits: raw positive or negative scores emitted by the final linear layer before normalization.",
          "Numerically stable Softmax: subtracting max(logits) before exp() so large numbers never overflow to NaN.",
          "Negative Log-Likelihood / Cross-Entropy Loss: -log(P(true_token))."
        ],
        code: `import torch, torch.nn.functional as F\n\nlogits = torch.tensor([[2.5, 0.5, -1.0]])\ntarget = torch.tensor([0])\n\nprobs = F.softmax(logits, dim=-1)\nloss = F.cross_entropy(logits, target)\nprint(f"Probs: {probs.tolist()} | Cross-Entropy Loss: {loss.item():.4f}")`,
        quiz: {
          question: "What input does PyTorch's F.cross_entropy(logits, targets) expect?",
          options: [
            "Probabilities that already passed through F.softmax()",
            "Raw unnormalized logits directly from the linear LM Head",
            "Binary 0/1 integers only"
          ],
          answerIndex: 1,
          explanation: "F.cross_entropy applies log_softmax internally for numerical stability, so passing Softmax probabilities double-normalizes them."
        },
        pitfall: "Passing Softmax probabilities into F.cross_entropy()! PyTorch's F.cross_entropy expects raw unnormalized logits.",
        project: "Build a Softmax & Cross-Entropy visualizer notebook showing how initial random LLM loss equals ln(vocab_size).",
        resourceTitle: "StatQuest — Cross-Entropy & Softmax Clearly Explained",
        resourceNote: "Intuitive walkthrough of log-probabilities and why Cross-Entropy penalizes confident wrong predictions heavily.",
        links: [
          { type: "YouTube", label: "StatQuest — Neural Networks Part 6: Cross Entropy", url: "https://www.youtube.com/watch?v=6ArSys5qHAU" },
          { type: "Community Guide", label: "Victor Zhou — Softmax & Cross-Entropy Loss Explained Simply", url: "https://victorzhou.com/blog/softmax/" }
        ]
      }
    ]
  },
  {
    id: "phase-2",
    number: "02",
    category: "math-pytorch",
    accent: "indigo",
    duration: "Week 3–4",
    title: "Neural Networks & Autograd from Scratch",
    subtitle: "Demystify 'magic' training loops by building a scalar backpropagation engine (micrograd) before graduating to PyTorch nn.Module.",
    plainBigPicture: "How does an AI actually 'learn' from its mistakes? Every calculation is recorded as a step-by-step recipe (a Computation Graph). When the model makes a wrong guess, 'Backpropagation' walks backward through that recipe using basic calculus to figure out which knobs ('Weights') to nudge slightly up or down.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 2 Skills",
      summary: "Every frontier model relies on reverse-mode automatic differentiation and gated MLP layers:",
      examples: [
        "Llama 3, Mistral, and Gemma replaced older ReLU layers with the exact SwiGLU gated MLP architecture you code in Module 2.3.",
        "AdamW with decoupled weight decay (Module 2.4) trained GPT-3, Llama 3, and DeepSeek models across GPU clusters."
      ]
    },
    jargonDecoder: [
      { term: "Weights", def: "Adjustable numbers inside the AI brain" },
      { term: "Gradient", def: "Direction to nudge a weight to reduce error" },
      { term: "Backpropagation", def: "Walking backward to assign blame for errors" },
      { term: "Autograd", def: "PyTorch's automatic gradient calculator" },
      { term: "AdamW", def: "Smart algorithm that updates the weights" }
    ],
    achievements: [
      "Explain exactly what happens inside loss.backward() because you built a working scalar Autograd engine (micrograd) yourself.",
      "Construct multi-layer neural networks with modern non-linear activations (GELU & SwiGLU) and proper weight initialization.",
      "Write bug-free PyTorch training loops using AdamW, zero_grad(), train()/eval() switching, and activation diagnostics."
    ],
    miniProject: "Code Micrograd from Scratch & Train a 2-Layer Neural Network Without PyTorch",
    topics: [
      {
        id: "p2-t1",
        codeNum: "2.1",
        animType: "autograd_graph",
        title: "Computation Graphs & The Chain Rule",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase2_autograd",
          fileName: "01_chain_rule_verify.py",
          steps: [
            "Create a new folder phase2_autograd/ inside VIT-Automation in VS Code.",
            "Create 01_chain_rule_verify.py inside phase2_autograd/ and paste the manual vs. finite-difference gradient checker.",
            "Run it in your (.venv) terminal to see analytical calculus match numerical nudges to 6 decimal places."
          ],
          starterCode: `# Verify manual Chain Rule derivative against numerical finite difference
def forward_loss(w, x=2.0, b=1.0, y=10.0):
    z = w * x + b
    diff = z - y
    return diff ** 2

w, x, b, y = 3.0, 2.0, 1.0, 10.0

# 1. Analytical Chain Rule gradient: dL/dw = dL/ddiff * ddiff/dz * dz/dw
z = w * x + b
diff = z - y
loss = diff ** 2
analytical_grad_w = (2 * diff) * 1.0 * x

# 2. Numerical finite-difference check: (f(w + h) - f(w)) / h
h = 1e-6
numerical_grad_w = (forward_loss(w + h) - forward_loss(w)) / h

print(f"Initial Loss:        {loss:.4f}")
print(f"Analytical dL/dw:    {analytical_grad_w:.6f}")
print(f"Numerical  dL/dw:    {numerical_grad_w:.6f}")`,
          runCmd: "python phase2_autograd/01_chain_rule_verify.py",
          expectedOutput: `Initial Loss:        9.0000
Analytical dL/dw:    -12.000000
Numerical  dL/dw:    -11.999996`
        },
        plainEnglish: "Imagine a chain of gears: turning Gear A turns Gear B twice as fast, and Gear B turns Gear C three times as fast—so Gear C moves 2 x 3 = 6 times as fast as Gear A. That multiplication of step-by-step sensitivities is the 'Chain Rule'.",
        liveExample: "PyTorch's dynamic autograd engine builds a DAG on every forward pass and traverses it backward via the Chain Rule.",
        jargonTerms: [
          { term: "Computation Graph", def: "A flowchart showing every add/multiply step from input to final error." },
          { term: "Derivative / Gradient", def: "How much the final error changes if you nudge one number slightly." },
          { term: "Chain Rule", def: "Multiplying step-by-step sensitivities along a chain of operations." }
        ],
        why: "Every neural network—from a 2-line equation to GPT-4—is a Directed Acyclic Graph (DAG) of elementary additions and multiplications differentiated via the Chain Rule.",
        lessonSections: [
          {
            heading: "1. Local Gradient vs. Upstream Gradient",
            body: "Each node in a computation graph only needs to know its own local derivative. Multiplying upstream gradient (dL/dout) by local derivative (dout/din) passes blame backward.",
            codeExample: "x, y, w, b = 2.0, 10.0, 3.0, 1.0\nz = w * x + b\nloss = (z - y) ** 2\ndloss_dz = 2 * (z - y)\ndloss_dw = dloss_dz * x"
          }
        ],
        concepts: [
          "Derivatives as local sensitivity: measuring how nudging an input by h changes the final loss.",
          "Decomposing complex expressions into intermediate nodes.",
          "The Chain Rule: multiplying upstream gradient (dL/dout) by local derivative (dout/din)."
        ],
        code: `x, y = 2.0, 10.0\nw, b = 3.0, 1.0\nz = w * x + b\ndiff = z - y\nloss = diff ** 2\n\ndloss_ddiff = 2 * diff\ndloss_dw = dloss_ddiff * x\nprint("Gradient wrt w:", dloss_dw)`,
        quiz: {
          question: "In L = (w * x + b - y)^2, if z = w * x + b and dL/dz = -6.0 while x = 2.0, what is dL/dw?",
          options: ["-12.0", "-3.0", "+6.0"],
          answerIndex: 0,
          explanation: "By the Chain Rule, dL/dw = (dL/dz) * (dz/dw) = -6.0 * x = -6.0 * 2.0 = -12.0."
        },
        pitfall: "Overwriting gradients (=) instead of accumulating them (+=) when a variable is used in multiple branches of a graph!",
        project: "Draw a 5-node computation graph on paper and verify your hand-calculated gradients against numerical finite differences.",
        resourceTitle: "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)",
        resourceNote: "Builds rock-solid geometric intuition for how gradients flow backward through expressions.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Backpropagation, Intuitively (Deep Learning Ch. 3)", url: "https://www.youtube.com/watch?v=Ilg3gGewQ5U" },
          { type: "Community Guide", label: "Chris Olah's Blog — Calculus on Computational Graphs: Backpropagation", url: "https://colah.github.io/posts/2015-08-Backprop/" }
        ]
      },
      {
        id: "p2-t2",
        codeNum: "2.2",
        animType: "autograd_graph",
        title: "Building micrograd: Scalar Autograd Engine",
        duration: "4 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase2_autograd",
          fileName: "02_micrograd_engine.py",
          steps: [
            "Create 02_micrograd_engine.py inside phase2_autograd/ in VS Code.",
            "Paste the complete working Value autograd class with topological sorting and backward() pass.",
            "Run it and verify that our pure-Python Value engine computes the exact same gradients as PyTorch!"
          ],
          starterCode: `import torch

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Test our pure-Python micrograd engine against PyTorch
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) * a
L.backward()

at = torch.tensor(2.0, requires_grad=True)
bt = torch.tensor(-3.0, requires_grad=True)
ct = torch.tensor(10.0, requires_grad=True)
Lt = (at * bt + ct) * at
Lt.backward()

print(f"micrograd grads -> a:{a.grad}, b:{b.grad}, c:{c.grad}")
print(f"PyTorch   grads -> a:{at.grad.item()}, b:{bt.grad.item()}, c:{ct.grad.item()}")`,
          runCmd: "python phase2_autograd/02_micrograd_engine.py",
          expectedOutput: `micrograd grads -> a:-2.0, b:4.0, c:2.0
PyTorch   grads -> a:-2.0, b:4.0, c:2.0`
        },
        plainEnglish: "When you call 'loss.backward()' in PyTorch, it feels like magic. Here you will build your own tiny 100-line clone of PyTorch's brain ('micrograd') using plain Python numbers.",
        liveExample: "PyTorch's C++ libtorch Autograd engine is a tensorized version of this exact Value DAG + topological reverse traversal.",
        jargonTerms: [
          { term: "Autograd", def: "Automatic differentiation—computing gradients automatically without hand-math." },
          { term: "Topological Sort", def: "Ordering graph steps so a node only runs backward after all nodes after it finish." }
        ],
        why: "Writing Karpathy's ~100-line micrograd engine removes all mystery from PyTorch's autograd.",
        lessonSections: [
          {
            heading: "1. Wrapping Floats in a Differentiable Value Object",
            body: "By overriding __add__ and __mul__ in Python, expressions like a * b + c automatically record their parent pointers and local backward closures.",
            codeExample: "class Value:\n    def __init__(self, data, _children=()):\n        self.data, self.grad = float(data), 0.0\n        self._backward = lambda: None\n        self._prev = set(_children)"
          }
        ],
        concepts: [
          "Wrapping floats in a Value class storing .data, .grad, ._prev children, and a ._backward() closure.",
          "Implementing __add__, __mul__, __pow__, tanh(), and relu() with local gradient rules.",
          "Topological sorting of the computation DAG so backward() visits every node in strict reverse dependency order."
        ],
        code: `class Value:\n    def __init__(self, data, _children=(), _op=''):\n        self.data, self.grad = float(data), 0.0\n        self._backward = lambda: None\n        self._prev = set(_children)\n\n    def __add__(self, other):\n        other = other if isinstance(other, Value) else Value(other)\n        out = Value(self.data + other.data, (self, other), '+')\n        def _backward():\n            self.grad += 1.0 * out.grad\n            other.grad += 1.0 * out.grad\n        out._backward = _backward\n        return out`,
        quiz: {
          question: "Why must backward() accumulate gradients with += instead of =?",
          options: [
            "Because a single weight or variable can be used in multiple branches of the graph",
            "To make floating point numbers larger",
            "Because Python doesn't support = inside closures"
          ],
          answerIndex: 0,
          explanation: "Multivariable calculus states that if variable x feeds two paths, its total gradient is the sum of gradients from both paths."
        },
        pitfall: "Forgetting to reset node.grad = 0.0 before each training iteration—accumulated gradients from previous steps will corrupt parameter updates.",
        project: "Implement the complete Value autograd engine in Python and verify its gradients against PyTorch's torch.autograd.",
        resourceTitle: "Andrej Karpathy — Building micrograd (Zero to Hero Ep. 1)",
        resourceNote: "Code along line-by-line in JupyterLab; this is the single highest-ROI exercise for LLM beginners.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — The spelled-out intro to neural networks: building micrograd", url: "https://www.youtube.com/watch?v=VMj-3S1tku0" },
          { type: "Community Guide", label: "Andrej Karpathy — Official micrograd GitHub Repository", url: "https://github.com/karpathy/micrograd" }
        ]
      },
      {
        id: "p2-t3",
        codeNum: "2.3",
        animType: "gradient_descent",
        title: "Neurons, Layers, MLPs & Non-Linear Activations (ReLU, GELU, SwiGLU)",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase2_autograd",
          fileName: "03_mlp_and_swiglu.py",
          steps: [
            "Create 03_mlp_and_swiglu.py inside phase2_autograd/ in VS Code.",
            "Paste the SwiGLU Feed-Forward Network implementation (used in Llama-3 & Gemma).",
            "Run it to verify parameter counts and output tensor shapes."
          ],
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

if __name__ == "__main__":
    mlp = SwiGLU(dim=128, hidden_dim=384)
    x = torch.randn(2, 10, 128)  # (Batch=2, Tokens=10, Dim=128)
    out = mlp(x)
    total_params = sum(p.numel() for p in mlp.parameters())
    print("SwiGLU input shape: ", x.shape)
    print("SwiGLU output shape:", out.shape)
    print("Total MLP params:   ", f"{total_params:,}")`,
          runCmd: "python phase2_autograd/03_mlp_and_swiglu.py",
          expectedOutput: `SwiGLU input shape:  torch.Size([2, 10, 128])
SwiGLU output shape: torch.Size([2, 10, 128])
Total MLP params:    147,456`
        },
        plainEnglish: "A single artificial 'Neuron' multiplies inputs by weights, adds them up, and passes the result through a non-linear curve ('Activation Function' like GELU or SwiGLU) so the network can learn complex patterns.",
        liveExample: "Around 2/3 of the 70 billion parameters in Llama-3 70B live inside its SwiGLU Feed-Forward MLP layers.",
        jargonTerms: [
          { term: "MLP", def: "Multi-Layer Perceptron—a stack of linear layers with activation curves between them." },
          { term: "Activation Function", def: "A non-linear bend (ReLU, GELU, SwiGLU) allowing AI to learn complex rules." }
        ],
        why: "Two-thirds of the parameters in a Transformer block live inside its Multi-Layer Perceptron (MLP), where non-linear activations unlock complex reasoning.",
        lessonSections: [
          {
            heading: "1. Why Non-Linearity Is Mandatory",
            body: "Without an activation function between two linear layers (x @ W1) @ W2, matrix associativity collapses them into a single linear matrix x @ (W1 @ W2), incapable of learning non-linear functions.",
            codeExample: "import torch.nn as nn, torch.nn.functional as F\n# SwiGLU gates an up-projection with a SiLU curve"
          }
        ],
        concepts: [
          "Single Neuron -> Layer of Neurons -> Multi-Layer Perceptron (MLP) architecture.",
          "Why non-linearities are essential: stacking linear layers without activations collapses into a single matrix multiply.",
          "Modern LLM activations: ReLU vs. smooth GELU (GPT-2/3) vs. gated SwiGLU (Llama 3 / Gemma)."
        ],
        code: `import torch, torch.nn as nn, torch.nn.functional as F\n\nclass SwiGLU(nn.Module):\n    def __init__(self, dim, hidden_dim):\n        super().__init__()\n        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)\n        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)\n        self.w_down = nn.Linear(hidden_dim, dim, bias=False)\n\n    def forward(self, x):\n        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))`,
        quiz: {
          question: "What happens if you stack 10 nn.Linear layers with NO activation functions between them?",
          options: [
            "It becomes 10x smarter than a Transformer",
            "Mathematically it collapses into a single linear matrix multiplication",
            "PyTorch raises a SyntaxError"
          ],
          answerIndex: 1,
          explanation: "Linear transformations composed together remain strictly linear; non-linear curves (GELU, SwiGLU) are required for universal function approximation."
        },
        pitfall: "Dead ReLU neurons: if weights initialize poorly or learning rate is too high, negative pre-activations output 0 forever.",
        project: "Build Neuron, Layer, and MLP classes on top of your micrograd Value class.",
        resourceTitle: "StatQuest & 3Blue1Brown — Neural Networks & Activations",
        resourceNote: "Visualizes how layered neurons compose edges and patterns into higher-level representations.",
        links: [
          { type: "YouTube", label: "StatQuest — The Essential Main Ideas of Neural Networks", url: "https://www.youtube.com/watch?v=CqOfi41LfDw" },
          { type: "Community Guide", label: "Papers With Code — SwiGLU & GELU Activation Functions Explained", url: "https://paperswithcode.com/method/swiglu" }
        ]
      },
      {
        id: "p2-t4",
        codeNum: "2.4",
        animType: "gradient_descent",
        title: "Gradient Descent, Learning Rate & AdamW Optimizer",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase2_autograd",
          fileName: "04_adamw_optimization.py",
          steps: [
            "Create 04_adamw_optimization.py inside phase2_autograd/ in VS Code.",
            "Paste the AdamW optimization script training a small linear model to learn a target rule.",
            "Run it and watch the MSE loss drop steadily every 25 steps."
          ],
          starterCode: `import torch
import torch.nn as nn

torch.manual_seed(42)
# Synthetic task: learn y = 3*x1 - 2*x2 + 0.5
X = torch.randn(64, 2)
true_w = torch.tensor([[3.0], [-2.0]])
Y = X @ true_w + 0.5

model = nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=0.01)

for step in range(1, 101):
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = nn.functional.mse_loss(pred, Y)
    loss.backward()
    optimizer.step()
    if step % 25 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.5f} | Learned W: {model.weight.data.tolist()[0]}")`,
          runCmd: "python phase2_autograd/04_adamw_optimization.py",
          expectedOutput: `Step  25 | Loss: 6.41283 | Learned W: [1.14, -0.92]
Step  50 | Loss: 1.82041 | Learned W: [2.11, -1.64]
Step  75 | Loss: 0.25190 | Learned W: [2.76, -1.93]
Step 100 | Loss: 0.01482 | Learned W: [2.95, -1.99]`
        },
        plainEnglish: "Imagine standing blindfolded on a mountain and trying to reach the lowest valley by feeling the slope under your feet. 'AdamW' is a smart GPS optimizer that adapts step sizes individually for every single weight.",
        liveExample: "Virtually every open-weights LLM (Llama, Mistral, Qwen, GPT-2) uses AdamW with betas=(0.9, 0.95) and decoupled weight decay.",
        jargonTerms: [
          { term: "Gradient Descent", def: "Stepping opposite to the gradient to steadily shrink prediction error." },
          { term: "Optimizer (AdamW)", def: "Smart algorithm that adjusts step sizes individually for every single weight." }
        ],
        why: "Plain SGD struggles on Transformers because embedding weights and attention projections have wildly different gradient scales. AdamW adapts step sizes per parameter.",
        lessonSections: [
          {
            heading: "1. The 3-Step PyTorch Update Cycle",
            body: "Always clear stale gradients with optimizer.zero_grad(set_to_none=True), backpropagate loss.backward(), and apply updates with optimizer.step().",
            codeExample: "optimizer.zero_grad(set_to_none=True)\nloss.backward()\noptimizer.step()"
          }
        ],
        concepts: [
          "Vanilla Gradient Descent update rule: param = param - lr * param.grad.",
          "Momentum and RMSProp adaptive scaling.",
          "AdamW: combining adaptive per-parameter learning rates with decoupled weight decay regularization."
        ],
        code: `import torch\n\nmodel = torch.nn.Linear(64, 64)\noptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)\nprint("AdamW initialized with lr=3e-4")`,
        quiz: {
          question: "Why is AdamW preferred over plain SGD for training Transformers?",
          options: [
            "AdamW adapts step sizes per parameter using gradient momentum and variance",
            "SGD doesn't work on GPUs",
            "AdamW requires zero memory"
          ],
          answerIndex: 0,
          explanation: "Transformer embeddings and attention projections experience vastly different gradient magnitudes, which AdamW normalizes per weight."
        },
        pitfall: "Using a huge learning rate like lr=0.1 with AdamW on a Transformer—LLM AdamW learning rates typically sit around 1e-4 to 6e-4.",
        project: "Compare SGD, SGD+Momentum, and AdamW on a noisy 2D bowl in Python.",
        resourceTitle: "StatQuest — Gradient Descent & Adam Optimizer Intuition",
        resourceNote: "Clear visual explanation of why momentum and adaptive learning rates speed up convergence.",
        links: [
          { type: "YouTube", label: "StatQuest — Gradient Descent, Step-by-Step", url: "https://www.youtube.com/watch?v=sDv4f4s2SB8" },
          { type: "Community Guide", label: "Distill.pub — Why Momentum Really Works (Interactive Visual Essay)", url: "https://distill.pub/2017/momentum/" }
        ]
      },
      {
        id: "p2-t5",
        codeNum: "2.5",
        animType: "gradient_descent",
        title: "Training Loop Anatomy, Initialization & Vanishing/Exploding Gradients",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase2_autograd",
          fileName: "05_init_and_eval_modes.py",
          steps: [
            "Create 05_init_and_eval_modes.py inside phase2_autograd/ in VS Code.",
            "Paste the GPT-style weight initialization & train()/eval() mode check script.",
            "Run it to verify that model.eval() eliminates Dropout randomness during inference."
          ],
          starterCode: `import torch
import torch.nn as nn

def init_gpt_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

block = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))
block.apply(init_gpt_weights)
x = torch.ones(1, 64)

block.train()
out_train_1 = block(x).sum().item()
out_train_2 = block(x).sum().item()

block.eval()
out_eval_1 = block(x).sum().item()
out_eval_2 = block(x).sum().item()

print(f"train() mode (Dropout ON):  run1={out_train_1:.3f}, run2={out_train_2:.3f}")
print(f"eval()  mode (Dropout OFF): run1={out_eval_1:.3f}, run2={out_eval_2:.3f}")`,
          runCmd: "python phase2_autograd/05_init_and_eval_modes.py",
          expectedOutput: `train() mode (Dropout ON):  run1=0.812, run2=-1.409
eval()  mode (Dropout OFF): run1=-0.214, run2=-0.214`
        },
        plainEnglish: "Before training starts, all weights are filled with tiny random numbers ('Initialization'). A clean 'Training Loop' repeats guess -> grade -> nudge thousands of times safely.",
        liveExample: "GPT-2 and GPT-3 scale residual output projection initialization by 1 / sqrt(2 * n_layers) so deep 96-layer stacks don't explode at step 0.",
        jargonTerms: [
          { term: "Initialization", def: "Setting smart starting random numbers in weights before step 1." },
          { term: "model.eval()", def: "Switches off training-only behaviors (like Dropout) when testing/chatting." }
        ],
        why: "If initial weights are too large, activations explode to NaN on step 1; if too small, gradients vanish to zero.",
        lessonSections: [
          {
            heading: "1. Switching Between train() and eval() Modes",
            body: "Dropout layers randomly zero out activations during training to prevent overfitting, so you must call model.eval() before generating text.",
            codeExample: "model.eval()\nwith torch.no_grad():\n    logits = model(tokens)"
          }
        ],
        concepts: [
          "The canonical PyTorch training loop: forward -> compute loss -> zero_grad -> backward -> clip_grad_norm -> step.",
          "GPT-style scaled normal initialization (std = 0.02).",
          "Diagnosing activation saturation and gradient norms across layers."
        ],
        code: `import torch.nn as nn\n\ndef init_weights(module):\n    if isinstance(module, nn.Linear):\n        nn.init.normal_(module.weight, mean=0.0, std=0.02)\n        if module.bias is not None:\n            nn.init.zeros_(module.bias)`,
        quiz: {
          question: "What happens if you forget to call model.eval() when generating text from a model trained with Dropout?",
          options: [
            "Dropout remains active and randomly zeroes out neurons during generation, degrading output quality",
            "The model runs twice as fast",
            "PyTorch refuses to run forward()"
          ],
          answerIndex: 0,
          explanation: "model.eval() disables Dropout randomness so inference is clean and deterministic."
        },
        pitfall: "Forgetting model.eval() during validation/inference and model.train() during training.",
        project: "Write a clean, reusable PyTorch trainer function with train/eval mode switching and weight initialization.",
        resourceTitle: "Andrej Karpathy — Building makemore Part 3: Activations & Gradients",
        resourceNote: "Deep diagnostic masterclass on inspecting activation histograms and gradient flow.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Building makemore Part 3: Activations & Gradients", url: "https://www.youtube.com/watch?v=P6sfmUTpUmc" },
          { type: "Community Guide", label: "Andrej Karpathy Blog — A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/" }
        ]
      }
    ]
  },
  {
    id: "phase-3",
    number: "03",
    category: "math-pytorch",
    accent: "teal",
    duration: "Week 5",
    title: "Text Tokenization & Data Pipelines",
    subtitle: "Convert raw text corpora into subword token IDs using Byte-Pair Encoding (BPE) and construct sliding-window PyTorch DataLoaders.",
    plainBigPicture: "Neural networks can only do math on numbers, never raw English letters. A 'Tokenizer' acts like a secret decoder ring: it chops text into common chunks ('Tokens') and assigns each chunk an integer ID.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 3 Skills",
      summary: "Tokenization directly explains famous quirks in ChatGPT and Claude:",
      examples: [
        "Why LLMs struggle to count how many 'r's are in 'strawberry': BPE compresses 'straw' and 'berry' into single integer IDs so the model never sees individual letters!",
        "GPT-4o expanded its BPE vocabulary from 50k to 200k tokens (o200k_base) to make multilingual & code generation 2x faster."
      ]
    },
    jargonDecoder: [
      { term: "Token", def: "A word or common word-piece (e.g., 'ing')" },
      { term: "Tokenizer", def: "Translates text <-> integer token IDs" },
      { term: "BPE", def: "Compression algorithm that merges frequent letter pairs" },
      { term: "Context Window", def: "Max tokens the model can read at once" },
      { term: "DataLoader", def: "Feeds shuffled training batches to the GPU" }
    ],
    achievements: [
      "Code a working Byte-Pair Encoding (BPE) tokenizer from scratch in pure Python (train, encode, and decode).",
      "Handle UTF-8 byte streams and special control tokens (<|endoftext|>, <|user|>, <|assistant|>) without crashes.",
      "Build a PyTorch sliding-window Dataset & DataLoader producing (X, Y) next-token training pairs."
    ],
    miniProject: "Code a Complete Byte-Pair Encoding (BPE) Tokenizer & Sliding-Window DataLoader",
    topics: [
      {
        id: "p3-t1",
        codeNum: "3.1",
        animType: "bpe_merge",
        title: "Character-Level vs. Word-Level vs. Subword Tokenization",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase3_tokenization",
          fileName: "01_compare_tokenizers.py",
          steps: [
            "Create a new folder phase3_tokenization/ inside VIT-Automation in VS Code.",
            "Create 01_compare_tokenizers.py and paste the sequence-length comparison script using Hugging Face tokenizers/tiktoken.",
            "Run it in your (.venv) terminal to compare character count vs. subword token count."
          ],
          starterCode: `from transformers import AutoTokenizer

# Load GPT-2's pretrained subword BPE tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

samples = [
    "The capital of France is Paris.",
    "unbelievable tokenization efficiency",
    "def forward(self, x): return x @ self.W"
]

for text in samples:
    token_ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"Chars: {len(text):2d} -> Tokens: {len(token_ids):2d} | Pieces: {pieces}")`,
          runCmd: "python phase3_tokenization/01_compare_tokenizers.py",
          expectedOutput: `Chars: 31 -> Tokens:  7 | Pieces: ['The', ' capital', ' of', ' France', ' is', ' Paris', '.']
Chars: 36 -> Tokens:  6 | Pieces: ['un', 'believ', 'able', ' token', 'ization', ' efficiency']
Chars: 39 -> Tokens: 12 | Pieces: ['def', ' forward', '(', 'self', ',', ' x', '):', ' return', ' x', ' @', ' self', '.W']`
        },
        plainEnglish: "Subword Tokenization splits rare words into reusable roots like 'un-' + 'believ' + '-able' while keeping frequent words intact as single IDs.",
        liveExample: "OpenAI's tiktoken and Meta's SentencePiece both use subword tokenization averaging ~0.75 English words per token.",
        jargonTerms: [
          { term: "Subword Token", def: "1 token = frequent word or root/suffix chunk (~0.75 words on average)." }
        ],
        why: "Tokenization defines the atomic symbols your LLM sees and predicts—directly governing context length efficiency, spelling, and code accuracy.",
        lessonSections: [
          {
            heading: "1. Why Subword Beats Character and Whole-Word",
            body: "Character-level makes sequences 5x longer (quadratic attention cost), while whole-word fails on typos and code identifiers. Subword BPE balances vocabulary size (~50k–128k) with compact sequences.",
            codeExample: "# 'unbelievable' -> ['un', 'believ', 'able'] (3 tokens instead of 12 chars)"
          }
        ],
        concepts: [
          "Character-level vs. Word-level vs. Subword tokenization (BPE / WordPiece / Unigram).",
          "Leading spaces in GPT tokenizers ('apple' vs ' apple')."
        ],
        code: `# "unbelievable" -> ["un", "believ", "able"] -> [403, 18924, 540]`,
        quiz: {
          question: "Why do modern LLMs use subword tokenization instead of pure character-level tokenization?",
          options: [
            "Character-level makes token sequences ~4-5x longer, dramatically increasing attention compute cost",
            "Characters cannot be represented as integers",
            "Subword tokenization eliminates the need for embeddings"
          ],
          answerIndex: 0,
          explanation: "Self-attention scales quadratically with sequence length T, so compressing 12 characters into 3 subwords speeds up training and inference massively."
        },
        pitfall: "Leading spaces matter! In GPT tokenizers, 'apple' and ' apple' have completely different token IDs.",
        project: "Compare sequence lengths across character-level, word-level, and subword tokenization.",
        resourceTitle: "Hugging Face LLM Course — Tokenizers & Karpathy's GPT Tokenizer",
        resourceNote: "Shows concrete examples of how tokenization quirks explain LLM failure modes.",
        links: [
          { type: "Community Guide", label: "OpenAI Tiktokenizer — Interactive Web Token Splitter", url: "https://tiktokenizer.vercel.app/" },
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer (Full Lecture)", url: "https://www.youtube.com/watch?v=zduSFxRajkE" }
        ]
      },
      {
        id: "p3-t2",
        codeNum: "3.2",
        animType: "bpe_merge",
        title: "Unicode, UTF-8 Byte Encoding & Vocabulary Trade-offs",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase3_tokenization",
          fileName: "02_utf8_byte_encoding.py",
          steps: [
            "Create 02_utf8_byte_encoding.py inside phase3_tokenization/ in VS Code.",
            "Paste the UTF-8 byte inspection script covering English, Tamil, Python code, and emojis.",
            "Run it to see how every string maps cleanly into integers 0..255."
          ],
          starterCode: `samples = [
    "Hello LLM",
    "வணக்கம் AI",  # Tamil script
    "Rocket 🚀"
]

for text in samples:
    utf8_bytes = list(text.encode("utf-8"))
    restored = bytes(utf8_bytes).decode("utf-8", errors="replace")
    print(f"Text: {text:14s} | Chars: {len(text):2d} | UTF-8 Bytes ({len(utf8_bytes):2d}): {utf8_bytes}")
    assert restored == text`,
          runCmd: "python phase3_tokenization/02_utf8_byte_encoding.py",
          expectedOutput: `Text: Hello LLM      | Chars:  9 | UTF-8 Bytes ( 9): [72, 101, 108, 108, 111, 32, 76, 76, 77]
Text: வணக்கம் AI     | Chars: 10 | UTF-8 Bytes (24): [224, 174, 181, ...]
Text: Rocket 🚀      | Chars:  8 | UTF-8 Bytes (11): [82, 111, 99, 107, 101, 116, 32, 240, 159, 154, 128]`
        },
        plainEnglish: "UTF-8 breaks any letter, Tamil/Hindi script, or emoji into 1 to 4 raw bytes (integers 0–255). Starting our tokenizer with those 256 base bytes guarantees 100% coverage of every language.",
        liveExample: "GPT-2, GPT-4, and Llama 3 all use Byte-Level BPE so they never emit <UNK> tokens on emojis or code.",
        jargonTerms: [
          { term: "UTF-8 Bytes", def: "Encoding any Unicode character into 1–4 byte integers (0 to 255)." }
        ],
        why: "Modern tokenizers operate on raw UTF-8 bytes (0–255) so every emoji, symbol, and language is representable without <UNK> tokens.",
        lessonSections: [
          {
            heading: "1. Converting Strings to Base Byte IDs",
            body: "In Python, list(text.encode('utf-8')) immediately converts any string into integers between 0 and 255.",
            codeExample: "raw_bytes = list('Hello 🚀'.encode('utf-8'))\nprint(raw_bytes)"
          }
        ],
        concepts: [
          "Unicode code points vs. UTF-8 variable-length byte encoding (1 to 4 bytes).",
          "Base vocabulary of 256 raw bytes guarantees 100% coverage."
        ],
        code: `text = "Hello 🚀 LLM"\nraw_bytes = list(text.encode("utf-8"))\nprint("UTF-8 byte IDs:", raw_bytes)\nprint("Decoded back:", bytes(raw_bytes).decode("utf-8"))`,
        quiz: {
          question: "How many base tokens does a Byte-Level BPE tokenizer start with before any merges?",
          options: ["26 (A-Z)", "256 (all possible byte values 0..255)", "50,257"],
          answerIndex: 1,
          explanation: "Any UTF-8 string is a sequence of bytes 0..255, giving a lossless 256-token base alphabet."
        },
        pitfall: "Always use .decode('utf-8', errors='replace') when decoding arbitrary generated IDs to avoid UnicodeDecodeError crashes.",
        project: "Write a helper script that inspects UTF-8 byte lengths across English, Tamil/Hindi, Python code, and emojis.",
        resourceTitle: "Andrej Karpathy — Let's Build the GPT Tokenizer (Unicode & UTF-8 Section)",
        resourceNote: "Explains why starting from UTF-8 bytes eliminates out-of-vocabulary errors forever.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE" },
          { type: "Blog / Docs", label: "Hugging Face LLM Course — Byte-Pair Encoding Tokenization", url: "https://huggingface.co/learn/llm-course/chapter6/5" }
        ]
      },
      {
        id: "p3-t3",
        codeNum: "3.3",
        animType: "bpe_merge",
        title: "Coding Byte-Pair Encoding (BPE) from Scratch (minbpe)",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase3_tokenization",
          fileName: "03_bpe_from_scratch.py",
          steps: [
            "Create 03_bpe_from_scratch.py inside phase3_tokenization/ in VS Code.",
            "Paste the complete working BasicBPETokenizer class (train, encode, and decode from scratch).",
            "Run it to watch BPE learn subword merges and verify round-trip lossless decoding."
          ],
          starterCode: `class BasicBPETokenizer:
    def __init__(self):
        self.merges = {}  # (p0, p1) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int):
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = self._get_stats(ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = self._merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

if __name__ == "__main__":
    corpus = "banana bandana banana bandana unbelievable tokenization " * 10
    tok = BasicBPETokenizer()
    tok.train(corpus, vocab_size=275)
    encoded = tok.encode("banana tokenization")
    print("Encoded IDs:", encoded)
    print("Decoded text:", tok.decode(encoded))`,
          runCmd: "python phase3_tokenization/03_bpe_from_scratch.py",
          expectedOutput: `Encoded IDs: [264, 32, 272, 105, 122, 97, 274]
Decoded text: banana tokenization`
        },
        plainEnglish: "BPE scans a corpus, finds the two adjacent tokens appearing most frequently, mints a brand-new token ID for that pair, and repeats until reaching the target vocabulary size.",
        liveExample: "Karpathy's minbpe replicates OpenAI's exact GPT-4 tokenizer algorithm in ~100 lines of readable Python.",
        jargonTerms: [
          { term: "BPE (Byte-Pair Encoding)", def: "Iteratively merging the most frequent adjacent token pair into one new token." },
          { term: "Merge Table", def: "The ordered rulebook of pair merges learned during tokenizer training." }
        ],
        why: "BPE is the exact algorithm used to train GPT-2, GPT-4, and Llama tokenizers.",
        lessonSections: [
          {
            heading: "1. The Pair-Merging Loop",
            body: "Each merge step replaces every occurrence of the top pair (p0, p1) with a new integer ID (256, 257, ...).",
            codeExample: "def merge_pair(ids, pair, new_id):\n    out, i = [], 0\n    while i < len(ids):\n        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n            out.append(new_id); i += 2\n        else:\n            out.append(ids[i]); i += 1\n    return out"
          }
        ],
        concepts: [
          "Counting all adjacent token pairs across the corpus with a frequency dictionary.",
          "Merging the highest-frequency pair into a new token ID and storing the merge rule.",
          "Encoding new text by applying learned merges in strict priority order."
        ],
        code: `def merge_pair(ids, pair, new_id):\n    out, i = [], 0\n    while i < len(ids):\n        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n            out.append(new_id)\n            i += 2\n        else:\n            out.append(ids[i])\n            i += 1\n    return out`,
        quiz: {
          question: "When encoding new text with a trained BPE tokenizer, in what order must merges be applied?",
          options: [
            "Alphabetical order",
            "In the exact priority order (lowest merge rank first) in which they were learned during training",
            "Random order"
          ],
          answerIndex: 1,
          explanation: "Earlier merges form sub-components required by later merges, so lowest-index merges must execute first."
        },
        pitfall: "In encode(), always apply the merge with the lowest merge index at each step—not arbitrary dictionary iteration order!",
        project: "Build a BasicTokenizer class with train(), encode(), and decode() and train a 512-token vocab on TinyShakespeare.",
        resourceTitle: "Andrej Karpathy — minbpe Repository & Tokenizer Lecture",
        resourceNote: "Walks through BasicTokenizer and RegexTokenizer step by step.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's Build the GPT Tokenizer", url: "https://www.youtube.com/watch?v=zduSFxRajkE" },
          { type: "Community Guide", label: "Andrej Karpathy — Official minbpe GitHub Repository", url: "https://github.com/karpathy/minbpe" }
        ]
      },
      {
        id: "p3-t4",
        codeNum: "3.4",
        animType: "bpe_merge",
        title: "Special Tokens (<|endoftext|>, PAD, BOS/EOS) & Context Windows",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase3_tokenization",
          fileName: "04_special_tokens_chatml.py",
          steps: [
            "Create 04_special_tokens_chatml.py inside phase3_tokenization/ in VS Code.",
            "Paste the ChatML special-token formatter script below.",
            "Run it to see how conversation boundaries are protected atomically."
          ],
          starterCode: `import re

SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,
    "<|user|>": 50257,
    "<|assistant|>": 50258,
}

def split_with_special_tokens(prompt: str):
    pattern = "(" + "|".join(re.escape(k) for k in SPECIAL_TOKENS) + ")"
    parts = [p for p in re.split(pattern, prompt) if p]
    return [(p, SPECIAL_TOKENS.get(p, "BPE_TEXT")) for p in parts]

chat = "<|user|>Explain DNS simply.<|assistant|>DNS translates domain names to IP addresses.<|endoftext|>"
for chunk, tag in split_with_special_tokens(chat):
    print(f"{str(tag):10s} | {chunk}")`,
          runCmd: "python phase3_tokenization/04_special_tokens_chatml.py",
          expectedOutput: `50257      | <|user|>
BPE_TEXT   | Explain DNS simply.
50258      | <|assistant|>
BPE_TEXT   | DNS translates domain names to IP addresses.
50256      | <|endoftext|>`
        },
        plainEnglish: "Special control tokens like <|endoftext|> and <|assistant|> act as structural delimiters so the model knows where documents end and whose turn it is in a chat.",
        liveExample: "ChatGPT's ChatML format uses <|im_start|>system, <|im_start|>user, and <|im_start|>assistant special token IDs.",
        jargonTerms: [
          { term: "Special Tokens", def: "Reserved control IDs like <|endoftext|> never split by BPE." },
          { term: "EOS", def: "End-of-Sequence stop token telling the model to stop generating." }
        ],
        why: "Special control tokens delimit separate documents during pretraining and separate User vs. Assistant turns during chat fine-tuning.",
        lessonSections: [
          {
            heading: "1. ChatML Conversation Delimiters",
            body: "Special tokens are matched atomically before regex/BPE splitting runs so they always map to a single dedicated integer ID.",
            codeExample: "special_tokens = {'<|endoftext|>': 50256, '<|user|>': 50257, '<|assistant|>': 50258}"
          }
        ],
        concepts: [
          "Document separators (<|endoftext|>) preventing unrelated articles from bleeding together.",
          "Conversation delimiters (<|im_start|>, <|im_end|>) and padding tokens.",
          "Context length (block_size / max_seq_len)."
        ],
        code: `special_tokens = {\n    "<|endoftext|>": 50256,\n    "<|user|>": 50257,\n    "<|assistant|>": 50258\n}`,
        quiz: {
          question: "How does an autoregressive LLM know when to stop generating a chat reply?",
          options: [
            "It always generates exactly 500 words",
            "It predicts the special End-of-Sequence (<|endoftext|> / <|im_end|>) token ID",
            "It stops at the first period"
          ],
          answerIndex: 1,
          explanation: "During SFT, assistant replies end with an EOS token; when sampled during inference, the generation loop halts."
        },
        pitfall: "Allowing regular BPE rules to split '<|endoftext|>' into separate characters.",
        project: "Extend your BPE tokenizer to register and encode/decode custom special tokens.",
        resourceTitle: "Sebastian Raschka — Chapter 2: Working with Text Data & Special Tokens",
        resourceNote: "Shows practical examples of handling unknown words and document boundary markers.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 2 Code Notebooks", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02" },
          { type: "Blog / Docs", label: "Hugging Face Docs — Chat Templates & Special Tokens", url: "https://huggingface.co/docs/transformers/main/en/chat_templating" }
        ]
      },
      {
        id: "p3-t5",
        codeNum: "3.5",
        animType: "sliding_window",
        title: "Sliding-Window Input/Target Pairs (X, Y) & PyTorch Dataset/DataLoader",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase3_tokenization",
          fileName: "05_sliding_window_dataloader.py",
          steps: [
            "Create 05_sliding_window_dataloader.py inside phase3_tokenization/ in VS Code.",
            "Paste the PyTorch GPTDataset & DataLoader code below.",
            "Run it to verify that Target Y is always Input X shifted forward by exactly 1 token."
          ],
          starterCode: `import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, token_ids: list[int], max_len: int, stride: int):
        self.inputs, self.targets = [], []
        for i in range(0, len(token_ids) - max_len, stride):
            self.inputs.append(torch.tensor(token_ids[i : i + max_len], dtype=torch.long))
            self.targets.append(torch.tensor(token_ids[i + 1 : i + max_len + 1], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

if __name__ == "__main__":
    dummy_corpus_ids = list(range(100, 200))
    ds = GPTDataset(dummy_corpus_ids, max_len=6, stride=6)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    print("Batch X shape:", x_batch.shape, "| Sample 0 X:", x_batch[0].tolist())
    print("Batch Y shape:", y_batch.shape, "| Sample 0 Y:", y_batch[0].tolist())`,
          runCmd: "python phase3_tokenization/05_sliding_window_dataloader.py",
          expectedOutput: `Batch X shape: torch.Size([2, 6]) | Sample 0 X: [100, 101, 102, 103, 104, 105]
Batch Y shape: torch.Size([2, 6]) | Sample 0 Y: [101, 102, 103, 104, 105, 106]`
        },
        plainEnglish: "LLMs learn via self-supervised fill-in-the-blank: Input X = [10, 20, 30] and Target Y = [20, 30, 40] (shifted 1 token right).",
        liveExample: "Pretraining pipelines pack trillions of web tokens into fixed-length blocks where every position predicts the next token simultaneously.",
        jargonTerms: [
          { term: "Input / Target (X, Y)", def: "Y is simply X shifted forward by 1 token (the 'next token' answer key)." },
          { term: "DataLoader", def: "PyTorch utility that packages samples and serves shuffled batches." }
        ],
        why: "A single sequence of T+1 tokens yields T self-supervised next-token prediction training examples in parallel.",
        lessonSections: [
          {
            heading: "1. Shifting Targets by One Position",
            body: "For any sequence slice of length T+1, x = chunk[:-1] and y = chunk[1:] creates T parallel next-token training targets.",
            codeExample: "x = torch.tensor(token_ids[i : i + max_len])\ny = torch.tensor(token_ids[i + 1 : i + max_len + 1])"
          }
        ],
        concepts: [
          "Shifting targets by 1 index: X = data[i : i+T], Y = data[i+1 : i+T+1].",
          "Stride parameter: overlapping vs. non-overlapping chunks.",
          "Wrapping token arrays in torch.utils.data.Dataset and DataLoader."
        ],
        code: `import torch\nfrom torch.utils.data import Dataset\n\nclass GPTDataset(Dataset):\n    def __init__(self, token_ids, max_len, stride):\n        self.inputs, self.targets = [], []\n        for i in range(0, len(token_ids) - max_len, stride):\n            self.inputs.append(torch.tensor(token_ids[i : i + max_len]))\n            self.targets.append(torch.tensor(token_ids[i + 1 : i + max_len + 1]))\n    def __len__(self): return len(self.inputs)\n    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]`,
        quiz: {
          question: "If Input X is [464, 3139, 286, 4881], what is the matching next-token Target Y?",
          options: ["[464, 3139, 286, 4881]", "[3139, 286, 4881, 318]", "[4881, 286, 3139, 464]"],
          answerIndex: 1,
          explanation: "Each position t in X predicts token t+1, so Y is shifted one index to the right."
        },
        pitfall: "Setting stride=1 on a large corpus creates near-duplicate overlapping windows that cause rapid memorization.",
        project: "Create a working GPTDataset and DataLoader over TinyShakespeare and verify batch shapes X: (32, 128) and Y: (32, 128).",
        resourceTitle: "PyTorch Datasets & DataLoaders + Sebastian Raschka Ch. 2.6",
        resourceNote: "Includes clear diagrams of input-target shifting and DataLoader batching.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Sliding Window DataLoader Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02" },
          { type: "Blog / Docs", label: "PyTorch Official Tutorials — Datasets & DataLoaders", url: "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html" }
        ]
      }
    ]
  },
  {
    id: "phase-4",
    number: "04",
    category: "transformer",
    accent: "indigo",
    duration: "Week 6–8",
    title: "Transformer Architecture & Self-Attention (The Heart of GPT)",
    subtitle: "Build every internal component of a Decoder-Only Transformer from scratch in pure PyTorch—from embeddings and QKV causal attention to stacked GPT blocks.",
    plainBigPicture: "This is the core engine behind ChatGPT, Claude, and Llama. First, token IDs become rich meaning vectors ('Embeddings'). Next, 'Self-Attention' lets words look back at earlier words for context. Stacking these blocks creates a GPT!",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 4 Skills",
      summary: "Every frontier model (GPT-4, Claude 3.5, Llama 3, DeepSeek) is built out of these exact Decoder-Only Transformer blocks:",
      examples: [
        "RoPE positional rotations (Module 4.1) enabled Llama 3 & Claude to scale context windows from 2K tokens up to 128K–200K tokens.",
        "Causal lower-triangular masking (Module 4.3) allows streaming token-by-token chat via KV Caching without recomputing past tokens."
      ]
    },
    jargonDecoder: [
      { term: "Embedding", def: "List of numbers capturing a word's meaning" },
      { term: "Self-Attention", def: "Words looking back at earlier context words" },
      { term: "Q, K, V", def: "Query (question), Key (label), Value (info)" },
      { term: "Causal Mask", def: "Blinder hiding future words during training" },
      { term: "LayerNorm", def: "Keeps numbers balanced across deep layers" }
    ],
    achievements: [
      "Implement Scaled Dot-Product Self-Attention and Multi-Head Attention with causal lower-triangular masking from scratch.",
      "Combine Token Embeddings, Positional Encodings (Learned & RoPE), LayerNorm, Residual Connections, and Feed-Forward MLPs.",
      "Assemble a complete, parameter-verified Decoder-Only GPT model in pure PyTorch ready for pretraining."
    ],
    miniProject: "Assemble a Complete Decoder-Only GPT Model in Pure PyTorch & Verify Forward Pass",
    topics: [
      {
        id: "p4-t1",
        codeNum: "4.1",
        animType: "embeddings_pos",
        title: "Token Embeddings & Positional Encodings (Learned vs. Sinusoidal vs. RoPE)",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "01_embeddings_and_positions.py",
          steps: [
            "Create a new folder phase4_transformer/ inside VIT-Automation in VS Code.",
            "Create 01_embeddings_and_positions.py and paste the Token + Positional Embedding module below.",
            "Run it in your (.venv) terminal to inspect how integer IDs turn into (B, T, emb_dim) tensors."
          ],
          starterCode: `import torch
import torch.nn as nn

B, T = 2, 8
vocab_size, emb_dim, max_len = 50257, 256, 512

tok_emb = nn.Embedding(vocab_size, emb_dim)
pos_emb = nn.Embedding(max_len, emb_dim)

token_ids = torch.randint(0, vocab_size, (B, T))
positions = torch.arange(T)

x = tok_emb(token_ids) + pos_emb(positions)
print("Token IDs shape:       ", token_ids.shape)
print("Combined input shape:  ", x.shape)
print("Embedding table params:", f"{tok_emb.weight.numel():,}")`,
          runCmd: "python phase4_transformer/01_embeddings_and_positions.py",
          expectedOutput: `Token IDs shape:        torch.Size([2, 8])
Combined input shape:   torch.Size([2, 8, 256])
Embedding table params: 12,865,792`
        },
        plainEnglish: "An 'Embedding' translates each token ID into a learned list of 256+ numbers where similar concepts sit close together, plus a 'Positional Encoding' stamp so word order matters.",
        liveExample: "Llama 3, Mistral, and Qwen use Rotary Position Embeddings (RoPE) so relative distance between words is encoded directly in Q and K dot products.",
        jargonTerms: [
          { term: "nn.Embedding", def: "A lookup table translating integer token IDs into rich decimal vectors." },
          { term: "RoPE", def: "Rotary Position Embedding—rotates vectors by position angle." }
        ],
        why: "Self-attention treats inputs as an unordered set; positional encodings inject word-order geometry.",
        lessonSections: [
          {
            heading: "1. Combining Semantic + Positional Vectors",
            body: "In GPT-2, token embeddings and position embeddings share the exact same dimension emb_dim and are added element-wise.",
            codeExample: "x = tok_emb(idx) + pos_emb(torch.arange(T))"
          }
        ],
        concepts: [
          "nn.Embedding(vocab_size, emb_dim): differentiable lookup table.",
          "Absolute Learned Positional Embeddings (GPT-2) vs. Sinusoidal vs. RoPE."
        ],
        code: `import torch, torch.nn as nn\n\nB, T, vocab_size, emb_dim, max_len = 2, 16, 50257, 256, 512\ntok_emb = nn.Embedding(vocab_size, emb_dim)\npos_emb = nn.Embedding(max_len, emb_dim)\nidx = torch.randint(0, vocab_size, (B, T))\nx = tok_emb(idx) + pos_emb(torch.arange(T))\nprint("Input embedding tensor shape:", x.shape)`,
        quiz: {
          question: "Why do Transformers require Positional Encodings?",
          options: [
            "Because self-attention is permutation-equivariant and has no built-in notion of token order",
            "To compress vocabulary size",
            "To replace Softmax"
          ],
          answerIndex: 0,
          explanation: "Without positional information, 'dog bites man' and 'man bites dog' produce identical attention sets."
        },
        pitfall: "Passing sequences longer than max_len into a learned positional embedding table.",
        project: "Plot the cosine similarity matrix of a learned positional embedding table before and after training.",
        resourceTitle: "Sebastian Raschka — Positional Embeddings Blog & Jay Alammar GPT-2",
        resourceNote: "Compare additive GPT-2 positional embeddings with modern multiplicative RoPE rotations.",
        links: [
          { type: "Community Guide", label: "Jay Alammar — The Illustrated GPT-2 (Embeddings & Decoder)", url: "https://jalammar.github.io/illustrated-gpt2/" },
          { type: "Community Guide", label: "EleutherAI Blog — Rotary Embeddings (RoPE)", url: "https://blog.eleuther.ai/rotary-embeddings/" }
        ]
      },
      {
        id: "p4-t2",
        codeNum: "4.2",
        animType: "qkv_attention",
        title: "Query, Key, Value (QKV) Projections & Scaled Dot-Product Attention",
        duration: "4 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "02_qkv_self_attention.py",
          steps: [
            "Create 02_qkv_self_attention.py inside phase4_transformer/ in VS Code.",
            "Paste the single-head QKV Scaled Dot-Product Attention module below.",
            "Run it and verify that every row of attention weights sums cleanly to 1.0."
          ],
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, head_dim: int):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, head_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, head_dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out, weights

if __name__ == "__main__":
    attn = SingleHeadAttention(emb_dim=64, head_dim=32)
    x = torch.randn(1, 4, 64)
    out, weights = attn(x)
    print("Output shape:", out.shape)
    print("Row sums of attention weights:", weights[0].sum(dim=-1).tolist())`,
          runCmd: "python phase4_transformer/02_qkv_self_attention.py",
          expectedOutput: `Output shape: torch.Size([1, 4, 32])
Row sums of attention weights: [1.0, 1.0, 1.0, 1.0]`
        },
        plainEnglish: "Every word emits a 'Query' (what I am looking for), a 'Key' (what label I advertise), and a 'Value' (the actual information I share if matched).",
        liveExample: "When you ask ChatGPT 'The trophy didn't fit in the suitcase because it was too small—what was small?', attention heads route 'it' strongly back to 'suitcase'.",
        jargonTerms: [
          { term: "Query (Q)", def: "Vector representing what context a token is currently searching for." },
          { term: "Key (K)", def: "Vector advertising what grammatical/semantic role a token holds." },
          { term: "Value (V)", def: "The actual content vector passed forward when Q matches K." }
        ],
        why: "Self-Attention lets each token dynamically query earlier tokens for context.",
        lessonSections: [
          {
            heading: "1. Why We Divide by sqrt(d_k)",
            body: "For large head dimensions d_k, dot products grow large in variance, pushing Softmax into near-zero gradients. Scaling by 1/sqrt(d_k) keeps variance at 1.",
            codeExample: "scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)"
          }
        ],
        concepts: [
          "Linear projections W_q, W_k, W_v.",
          "Raw affinity scores: Q @ K.transpose(-2, -1) / sqrt(d_k).",
          "Weighted sum of Value vectors via Softmax."
        ],
        code: `import torch, torch.nn.functional as F\n\nB, T, d_k = 2, 6, 32\nQ, K, V = torch.randn(B, T, d_k), torch.randn(B, T, d_k), torch.randn(B, T, d_k)\nscores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)\nattn_weights = F.softmax(scores, dim=-1)\ncontext = attn_weights @ V\nprint("Context shape:", context.shape)`,
        quiz: {
          question: "Why do we divide Q @ K^T by sqrt(d_k) before applying Softmax?",
          options: [
            "To prevent large dot products from pushing Softmax into saturated regions with vanishing gradients",
            "To double the sequence length",
            "To convert float32 into int64"
          ],
          answerIndex: 0,
          explanation: "Scaling by 1/sqrt(d_k) normalizes dot-product variance back to ~1 regardless of head dimension."
        },
        pitfall: "Forgetting to divide by sqrt(d_k)!",
        project: "Implement a single-head SelfAttention class in PyTorch and verify that every row of attn_weights sums to 1.0.",
        resourceTitle: "3Blue1Brown — Attention in Transformers, Visually Explained",
        resourceNote: "Best geometric animation on the internet for understanding Q, K, V dot products.",
        links: [
          { type: "YouTube", label: "3Blue1Brown — Attention in Transformers, Step-by-Step (Ch. 6)", url: "https://www.youtube.com/watch?v=eMlx5fFNoYc" },
          { type: "Community Guide", label: "Harvard NLP — The Annotated Transformer", url: "https://nlp.seas.harvard.edu/annotated-transformer/" }
        ]
      },
      {
        id: "p4-t3",
        codeNum: "4.3",
        animType: "causal_mask",
        title: "Causal Masking (tril) for Autoregressive Next-Token Prediction",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "03_causal_mask_tril.py",
          steps: [
            "Create 03_causal_mask_tril.py inside phase4_transformer/ in VS Code.",
            "Paste the lower-triangular causal mask script below.",
            "Run it to see all upper-right future entries become exact 0.000 after Softmax."
          ],
          starterCode: `import torch

T = 4
scores = torch.randn(T, T)
mask = torch.tril(torch.ones(T, T))
masked_scores = scores.masked_fill(mask == 0, float("-inf"))
probs = torch.softmax(masked_scores, dim=-1)

print("Causal Attention Matrix (future tokens = 0.000):")
print(torch.round(probs, decimals=3))`,
          runCmd: "python phase4_transformer/03_causal_mask_tril.py",
          expectedOutput: `Causal Attention Matrix (future tokens = 0.000):
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4120, 0.5880, 0.0000, 0.0000],
        [0.2190, 0.3410, 0.4400, 0.0000],
        [0.1850, 0.2740, 0.3100, 0.2310]])`
        },
        plainEnglish: "Without a 'Causal Mask', word #3 could cheat by peeking ahead at word #4 during parallel training! We fill the upper-right triangle with -inf so future words get 0% attention.",
        liveExample: "Decoder-only LLMs (GPT, Claude, Llama) use causal masking so all T token positions train simultaneously without future leakage.",
        jargonTerms: [
          { term: "Causal Mask", def: "Triangular barrier blocking tokens from peeking at future tokens." },
          { term: "torch.tril", def: "'Triangle Lower'—keeps the bottom-left triangle and zeroes out top-right." }
        ],
        why: "Causal masking prevents token t from peeking at future tokens t+1..T so parallel training matches sequential inference.",
        lessonSections: [
          {
            heading: "1. Masking with -inf Before Softmax",
            body: "Because exp(-inf) = 0, masking upper-triangle scores with float('-inf') before Softmax guarantees exact 0.0 weights on future positions while keeping row sums at 1.0.",
            codeExample: "masked = scores.masked_fill(mask == 0, float('-inf'))\nprobs = torch.softmax(masked, dim=-1)"
          }
        ],
        concepts: [
          "Lower-triangular matrix torch.tril(torch.ones(T, T)).",
          "Masking future positions with -inf BEFORE Softmax.",
          "Registering the mask via self.register_buffer('mask', ...)."
        ],
        code: `import torch\nT = 4\nscores = torch.randn(T, T)\nmask = torch.tril(torch.ones(T, T))\nmasked_scores = scores.masked_fill(mask == 0, float('-inf'))\nprint(torch.softmax(masked_scores, dim=-1))`,
        quiz: {
          question: "Why must future attention scores be set to -inf BEFORE Softmax rather than 0 AFTER Softmax?",
          options: [
            "Because exp(-inf) = 0, ensuring future tokens get 0% weight AND the remaining past weights still sum cleanly to 1.0",
            "Because PyTorch doesn't allow 0 inside tensors",
            "To make the matrix symmetric"
          ],
          answerIndex: 0,
          explanation: "Zeroing probabilities after Softmax would leave rows summing to less than 1.0, distorting activations."
        },
        pitfall: "Masking with 0 AFTER Softmax instead of -inf BEFORE Softmax!",
        project: "Add causal masking with register_buffer to your SelfAttention module.",
        resourceTitle: "Andrej Karpathy — Let's Build GPT (The Mathematical Trick in Self-Attention)",
        resourceNote: "Walks through weighted aggregation of past tokens using lower-triangular matrices.",
        links: [
          { type: "YouTube", label: "Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY" },
          { type: "Community Guide", label: "Sebastian Raschka — Understanding & Coding Self-Attention From Scratch", url: "https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html" }
        ]
      },
      {
        id: "p4-t4",
        codeNum: "4.4",
        animType: "qkv_attention",
        title: "Multi-Head Self-Attention & Tensor Reshaping / Concatenation",
        duration: "4 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "04_multi_head_attention.py",
          steps: [
            "Create 04_multi_head_attention.py inside phase4_transformer/ in VS Code.",
            "Paste the complete batched CausalMultiHeadAttention PyTorch class below.",
            "Run it to test parallel multi-head attention with register_buffer causal masking."
          ],
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, max_len: int = 512):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

if __name__ == "__main__":
    mha = CausalMultiHeadAttention(emb_dim=128, n_heads=4)
    x = torch.randn(2, 12, 128)
    print("Multi-Head Attention output shape:", mha(x).shape)`,
          runCmd: "python phase4_transformer/04_multi_head_attention.py",
          expectedOutput: `Multi-Head Attention output shape: torch.Size([2, 12, 128])`
        },
        plainEnglish: "'Multi-Head Attention' splits the embedding into 6 to 32 parallel spotlights ('Heads') so different heads can track grammar, pronouns, and punctuation simultaneously.",
        liveExample: "GPT-3 uses 96 parallel attention heads per layer; modern models like Llama-3 use Grouped-Query Attention (GQA) to share K/V heads for faster inference.",
        jargonTerms: [
          { term: "Attention Head", def: "An independent QKV spotlight focusing on one type of word relationship." },
          { term: ".contiguous()", def: "Re-aligns tensor memory layout after swapping axes so .view() works." }
        ],
        why: "Multiple heads let the model attend simultaneously to syntactic structure, entity references, and local phrasing across distinct subspaces.",
        lessonSections: [
          {
            heading: "1. Batched 4D Head Splitting",
            body: "Reshaping (B, T, C) -> (B, T, n_heads, head_dim) and transposing to (B, n_heads, T, head_dim) executes all attention heads in a single batched GPU kernel.",
            codeExample: "q = q.view(B, T, n_heads, head_dim).transpose(1, 2)"
          }
        ],
        concepts: [
          "Splitting C = n_heads * head_dim.",
          "Batched 4D tensor reshape and transpose.",
          "Merging heads back via .transpose(1, 2).contiguous().view(B, T, C)."
        ],
        code: `import torch.nn as nn\n\nclass MultiHeadAttention(nn.Module):\n    def __init__(self, emb_dim, n_heads):\n        super().__init__()\n        self.n_heads, self.head_dim = n_heads, emb_dim // n_heads\n        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)\n        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)\n\n    def forward(self, x):\n        B, T, C = x.shape\n        q, k, v = self.qkv(x).chunk(3, dim=-1)\n        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n        return self.proj(q.transpose(1, 2).contiguous().view(B, T, C))`,
        quiz: {
          question: "Why is .contiguous() required before .view(B, T, C) after calling .transpose(1, 2)?",
          options: [
            "Because .transpose() alters tensor strides without moving memory, and .view() requires contiguous memory",
            "To move the tensor to CPU",
            "To apply LayerNorm automatically"
          ],
          answerIndex: 0,
          explanation: "Transposing creates a non-contiguous view; calling .contiguous() packs memory linearly so .view() can merge head dimensions."
        },
        pitfall: "Forgetting .contiguous() before .view(B, T, C) after .transpose(1, 2).",
        project: "Code a complete batched MultiHeadAttention module in PyTorch.",
        resourceTitle: "Sebastian Raschka — Multi-Head Attention & Vizuara SLM Walkthrough",
        resourceNote: "Compares naive stacked heads with efficient batched weight splits.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 3.5 Multi-Head Attention Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03" },
          { type: "YouTube", label: "Vizuara — Build a Small Language Model (SLM) From Scratch", url: "https://www.youtube.com/watch?v=pOFcwcwtv3k" }
        ]
      },
      {
        id: "p4-t5",
        codeNum: "4.5",
        animType: "transformer_block",
        title: "LayerNorm / RMSNorm, Residual Connections & Feed-Forward MLP Blocks",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "05_layernorm_and_block.py",
          steps: [
            "Create 05_layernorm_and_block.py inside phase4_transformer/ in VS Code.",
            "Paste the custom LayerNorm implementation and compare it against PyTorch's built-in nn.LayerNorm.",
            "Run it to verify numerical equivalence to 1e-5."
          ],
          starterCode: `import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta

if __name__ == "__main__":
    x = torch.randn(2, 5, 64)
    custom_ln = CustomLayerNorm(64)
    torch_ln = nn.LayerNorm(64)
    max_diff = (custom_ln(x) - torch_ln(x)).abs().max().item()
    print(f"Max difference vs nn.LayerNorm: {max_diff:.8f}")`,
          runCmd: "python phase4_transformer/05_layernorm_and_block.py",
          expectedOutput: `Max difference vs nn.LayerNorm: 0.00000048`
        },
        plainEnglish: "'LayerNorm' re-centers every token's numbers to a clean bell curve, and 'Residual Connections' (x = x + layer(x)) act like an express elevator carrying gradients cleanly top-to-bottom.",
        liveExample: "Llama 3 and Gemma use Pre-Norm with RMSNorm—simplifying LayerNorm by dropping mean subtraction for a 10–15% speedup.",
        jargonTerms: [
          { term: "LayerNorm / RMSNorm", def: "Re-scales each token's numbers to mean 0, std 1 so deep layers stay stable." },
          { term: "Residual Connection", def: "Adding the input directly to the output (x + block(x)) as a gradient highway." }
        ],
        why: "Deep Transformers only train stably because residual highways let gradients flow unimpeded from loss straight back to early embeddings.",
        lessonSections: [
          {
            heading: "1. Pre-Norm Residual Stream",
            body: "Modern Transformers normalize BEFORE attention and MLP (x + Attn(Norm(x))), keeping an unobstructed identity path from loss to embeddings.",
            codeExample: "x = x + self.attn(self.ln1(x))\nx = x + self.mlp(self.ln2(x))"
          }
        ],
        concepts: [
          "Pre-Norm Transformer Block: x = x + Attention(Norm(x)), then x = x + MLP(Norm(x)).",
          "LayerNorm vs. RMSNorm.",
          "Position-wise Feed-Forward MLP: expanding C -> 4*C -> C."
        ],
        code: `import torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(cfg.emb_dim)\n        self.attn = MultiHeadAttention(cfg.emb_dim, cfg.n_heads)\n        self.ln2 = nn.LayerNorm(cfg.emb_dim)\n        self.mlp = nn.Sequential(\n            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),\n            nn.GELU(),\n            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),\n        )\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        x = x + self.mlp(self.ln2(x))\n        return x`,
        quiz: {
          question: "How do Residual Connections (x + Sublayer(x)) help train deep 96-layer Transformers?",
          options: [
            "They create an unobstructed highway for gradients to flow backward without vanishing",
            "They reduce vocabulary size by half",
            "They eliminate the need for a tokenizer"
          ],
          answerIndex: 0,
          explanation: "Because d(x + f(x))/dx = 1 + df/dx, gradients always have a clean '+1' path straight back to early layers."
        },
        pitfall: "Using BatchNorm instead of LayerNorm/RMSNorm!",
        project: "Implement LayerNorm from scratch and verify it matches nn.LayerNorm to 1e-5.",
        resourceTitle: "Sebastian Raschka — Chapter 4: Implementing a GPT Model from Scratch",
        resourceNote: "Implements LayerNorm, GELU, shortcut connections, and the full TransformerBlock cleanly.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 4 TransformerBlock Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04" },
          { type: "Community Guide", label: "Pinecone Learning Center — Layer Normalization in Transformers Explained", url: "https://www.pinecone.io/learn/batch-layer-normalization/" }
        ]
      },
      {
        id: "p4-t6",
        codeNum: "4.6",
        animType: "transformer_block",
        title: "Assembling the Full Decoder-Only GPT Architecture & Parameter Counting",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase4_transformer",
          fileName: "06_full_gpt_model.py",
          steps: [
            "Create 06_full_gpt_model.py inside phase4_transformer/ in VS Code.",
            "Paste the complete end-to-end Decoder-Only GPTModel script below.",
            "Run it to verify forward pass logits shape (B, T, vocab_size) and exact parameter count."
          ],
          starterCode: `from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.emb_dim)
        self.attn = nn.MultiheadAttention(cfg.emb_dim, cfg.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPTModel(cfg)
    dummy_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(dummy_ids)
    total_params = sum(p.numel() for p in model.parameters())
    print("GPT Logits shape:", logits.shape)
    print("Total unique parameters:", f"{total_params:,}")`,
          runCmd: "python phase4_transformer/06_full_gpt_model.py",
          expectedOutput: `GPT Logits shape: torch.Size([2, 16, 50257])
Total unique parameters: 30,025,344`
        },
        plainEnglish: "Now we snap all the Lego bricks together: token embeddings + positional embeddings -> N stacked TransformerBlocks -> final LayerNorm -> LM Head.",
        liveExample: "GPT-2 Small (124M params), Gemma 2B, and Qwen 2.5 follow this exact end-to-end Decoder-Only blueprint.",
        jargonTerms: [
          { term: "nn.ModuleList", def: "PyTorch's official list wrapper so stacked layers register properly for GPU & training." },
          { term: "Weight Tying", def: "Reusing the input embedding matrix as the output LM Head to save millions of params." }
        ],
        why: "Connecting embeddings, N Transformer blocks, final LayerNorm, and the Language Model Head gives you a genuine GPT architecture.",
        lessonSections: [
          {
            heading: "1. Weight Tying Between Embedding and LM Head",
            body: "Sharing self.tok_emb.weight = self.lm_head.weight saves vocab_size * emb_dim parameters (38.6M params in GPT-2!) and ties input/output semantic geometry.",
            codeExample: "self.tok_emb.weight = self.lm_head.weight"
          }
        ],
        concepts: [
          "Stacking N identical TransformerBlocks via nn.ModuleList.",
          "Final LayerNorm + Linear LM Head projecting (B, T, C) -> (B, T, vocab_size).",
          "Weight tying to save millions of parameters."
        ],
        code: `import torch.nn as nn\n\nclass GPTModel(nn.Module):\n    def __init__(self, cfg):\n        super().__init__()\n        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)\n        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)\n        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])\n        self.ln_f = nn.LayerNorm(cfg.emb_dim)\n        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)\n        self.tok_emb.weight = self.lm_head.weight`,
        quiz: {
          question: "What happens if you store Transformer blocks in a plain Python list [] instead of nn.ModuleList([])?",
          options: [
            "PyTorch won't register their parameters, so model.parameters() and .to('cuda') will silently skip them",
            "Training runs 2x faster",
            "It automatically enables LoRA"
          ],
          answerIndex: 0,
          explanation: "nn.Module only tracks submodules stored in attributes or PyTorch containers like nn.ModuleList / nn.Sequential."
        },
        pitfall: "Storing Transformer blocks in a plain Python list instead of nn.ModuleList!",
        project: "Instantiate a ~10M-parameter mini-GPT and print exact parameter counts per component.",
        resourceTitle: "Andrej Karpathy nanoGPT & freeCodeCamp Full LLM Build",
        resourceNote: "Complete end-to-end GPTModel walkthrough with parameter breakdown.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — Official nanoGPT Repository (model.py)", url: "https://github.com/karpathy/nanoGPT" },
          { type: "Community Guide", label: "Brendan Bycroft — LLM Visualization (Interactive 3D GPT Walkthrough)", url: "https://bbycroft.net/llm" }
        ]
      }
    ]
  },
  {
    id: "phase-5",
    number: "05",
    category: "training",
    accent: "amber",
    duration: "Week 9–10",
    title: "Pretraining, Evaluation & Text Generation",
    subtitle: "Train your custom GPT from random weights on real text, track loss & perplexity curves, and decode coherent completions.",
    plainBigPicture: "A freshly built GPT starts with random weights—so its first output is pure gibberish. During 'Pretraining', we feed it real text, measure its surprise ('Perplexity'), warm up and decay its learning rate, and watch coherent English emerge!",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 5 Skills",
      summary: "Every API slider and training stability trick in frontier AI labs comes directly from Phase 5:",
      examples: [
        "The 'Temperature' and 'Top-P' parameters in OpenAI, Anthropic, and Ollama APIs execute the exact logit scaling & nucleus sorting code in Module 5.5.",
        "Gradient clipping (max_norm=1.0) and Cosine LR Warmup (Modules 5.2–5.3) keep multi-month training runs from diverging."
      ]
    },
    jargonDecoder: [
      { term: "Pretraining", def: "Teaching a random model language from raw text" },
      { term: "Perplexity", def: "How confused the model is (1 = certain)" },
      { term: "Overfitting", def: "Memorizing training text instead of generalizing" },
      { term: "Temperature", def: "Creativity dial when picking the next word" },
      { term: "Top-k / Top-p", def: "Filtering out weird low-probability words" }
    ],
    achievements: [
      "Pretrain your custom GPT from random weights on real text using gradient accumulation, gradient clipping, and cosine LR warmup.",
      "Measure model uncertainty via Perplexity and diagnose overfitting vs. underfitting from train/validation loss curves.",
      "Code an autoregressive text generator supporting Greedy, Temperature scaling, Top-k, and Top-p (Nucleus) sampling."
    ],
    miniProject: "Pretrain a Mini-GPT on TinyShakespeare & Generate Text with Temperature/Top-k/Top-p",
    topics: [
      {
        id: "p5-t1",
        codeNum: "5.1",
        animType: "softmax_logits",
        title: "Next-Token Prediction Objective & Perplexity Metric",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase5_pretraining",
          fileName: "01_perplexity_eval.py",
          steps: [
            "Create a new folder phase5_pretraining/ inside VIT-Automation in VS Code.",
            "Create 01_perplexity_eval.py and paste the 3D-to-2D Cross-Entropy & Perplexity calculation script.",
            "Run it to see how loss translates into perplexity."
          ],
          starterCode: `import torch
import torch.nn.functional as F

B, T, V = 2, 10, 50257
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))

with torch.no_grad():
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
    perplexity = torch.exp(loss)

print(f"Cross-Entropy Loss: {loss.item():.4f}")
print(f"Perplexity:         {perplexity.item():,.1f} effective word choices")`,
          runCmd: "python phase5_pretraining/01_perplexity_eval.py",
          expectedOutput: `Cross-Entropy Loss: 11.3184
Perplexity:         82,322.1 effective word choices`
        },
        plainEnglish: "Taking e^loss gives 'Perplexity'—which means the model is roughly as uncertain as rolling an N-sided die to guess the next word. Lower perplexity = smarter language model!",
        liveExample: "GPT-2 improved zero-shot WikiText perplexity from ~65 down to ~18 as parameter scale and data quality increased.",
        jargonTerms: [
          { term: "Perplexity", def: "exp(loss)—effective number of equally likely next words the model considers." }
        ],
        why: "Understanding how 3D logits (B, T, V) flatten into 2D cross-entropy inputs—and how Perplexity translates loss—is core to LLM evaluation.",
        lessonSections: [
          {
            heading: "1. Flattening (B, T, V) for Cross-Entropy",
            body: "PyTorch's F.cross_entropy expects 2D predictions (N, Classes) and 1D targets (N,), so we flatten B and T together via .view(B * T, V).",
            codeExample: "loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))\nppl = torch.exp(loss)"
          }
        ],
        concepts: [
          "Flattening logits from (B, T, vocab_size) -> (B*T, vocab_size).",
          "Perplexity = exp(loss).",
          "Evaluating loss on a held-out validation split with torch.no_grad()."
        ],
        code: `import torch, torch.nn.functional as F\nB, T, V = 2, 8, 50257\nlogits = torch.randn(B, T, V)\ntargets = torch.randint(0, V, (B, T))\nloss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))\nprint(f"Loss: {loss.item():.3f} | Perplexity: {torch.exp(loss).item():.2f}")`,
        quiz: {
          question: "If a language model achieves a validation Cross-Entropy loss of 2.0, what is its Perplexity?",
          options: ["e^2.0 ≈ 7.39", "2.0", "0.5"],
          answerIndex: 0,
          explanation: "Perplexity is defined as exp(Cross-Entropy Loss), meaning the model is as uncertain as choosing uniformly among ~7.4 words."
        },
        pitfall: "Computing validation loss with gradients enabled—always wrap evaluation loops in 'with torch.no_grad():'.",
        project: "Write an evaluate_loss(model, val_loader) helper that reports average validation loss and perplexity.",
        resourceTitle: "Hugging Face Transformers Docs — Perplexity of Fixed-Length Models",
        resourceNote: "Explains the mathematical link between Cross-Entropy loss and Perplexity.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5 Pretraining & Evaluation Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" },
          { type: "Blog / Docs", label: "Hugging Face Docs — Perplexity of Fixed-Length Models", url: "https://huggingface.co/docs/transformers/perplexity" }
        ]
      },
      {
        id: "p5-t2",
        codeNum: "5.2",
        animType: "gradient_descent",
        title: "Pretraining Loop on TinyShakespeare: Gradient Accumulation & Clipping",
        duration: "4 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase5_pretraining",
          fileName: "02_pretrain_mini_gpt.py",
          steps: [
            "Create 02_pretrain_mini_gpt.py inside phase5_pretraining/ in VS Code.",
            "Paste the self-contained mini-GPT pretraining loop with gradient accumulation, clipping, and checkpoint saving.",
            "Run it in your (.venv) terminal and watch loss fall rapidly in seconds."
          ],
          starterCode: `import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
text = "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer. " * 40
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

class TinyLM(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.head(out)

model = TinyLM(len(chars))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
grad_accum_steps = 2

for step in range(1, 121):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum_steps):
        ix = torch.randint(0, len(data) - 33, (16,))
        xb = torch.stack([data[i : i + 32] for i in ix])
        yb = torch.stack([data[i + 1 : i + 33] for i in ix])
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1)) / grad_accum_steps
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if step % 40 == 0:
        print(f"Step {step:3d} | Effective Loss: {loss.item() * grad_accum_steps:.4f}")

torch.save(model.state_dict(), "phase5_pretraining/tiny_ckpt.pt")
print("Saved checkpoint -> phase5_pretraining/tiny_ckpt.pt")`,
          runCmd: "python phase5_pretraining/02_pretrain_mini_gpt.py",
          expectedOutput: `Step  40 | Effective Loss: 0.7412
Step  80 | Effective Loss: 0.1385
Step 120 | Effective Loss: 0.0894
Saved checkpoint -> phase5_pretraining/tiny_ckpt.pt`
        },
        plainEnglish: "'Gradient Accumulation' runs several small mini-batches and adds up their gradients before stepping! 'Gradient Clipping' caps giant gradient spikes so one weird sentence never blows up your weights.",
        liveExample: "Llama 3 trained with global batch sizes of 4M–16M tokens using gradient accumulation across micro-batches.",
        jargonTerms: [
          { term: "Gradient Accumulation", def: "Summing gradients across several mini-batches to simulate a big batch." },
          { term: "Gradient Clipping", def: "Capping giant gradient spikes (max_norm=1.0) to prevent training crashes." }
        ],
        why: "Gradient accumulation lets your laptop/GPU simulate large batches without running out of VRAM.",
        lessonSections: [
          {
            heading: "1. Scaling Loss by grad_accum_steps",
            body: "Because PyTorch sums gradients across .backward() calls, divide loss by grad_accum_steps so the accumulated gradient equals the true mean.",
            codeExample: "(loss / grad_accum_steps).backward()"
          }
        ],
        concepts: [
          "Gradient accumulation across micro-batches.",
          "Gradient norm clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).",
          "Checkpointing model + optimizer state dicts."
        ],
        code: `import torch\n# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n# torch.save({"model": model.state_dict(), "step": 100}, "ckpt.pt")\nprint("Gradient clipping & checkpoint pattern ready")`,
        quiz: {
          question: "When accumulating gradients over 4 micro-batches before calling optimizer.step(), how should you scale each micro-batch loss?",
          options: [
            "Divide loss by 4 before calling loss.backward()",
            "Multiply loss by 4",
            "Call optimizer.zero_grad() after every micro-batch"
          ],
          answerIndex: 0,
          explanation: "Dividing by grad_accum_steps ensures the summed gradients match the true average gradient of a 4x larger batch."
        },
        pitfall: "Forgetting to divide loss by grad_accum_steps before calling loss.backward()!",
        project: "Pretrain your mini-GPT on TinyShakespeare for 3,000 steps and watch validation loss drop below ~1.6.",
        resourceTitle: "Andrej Karpathy — nanoGPT Training Loop & TinyShakespeare Dataset",
        resourceNote: "Shows real-time loss progression and sample text evolving from gibberish to Shakespearean dialogue.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — nanoGPT train.py", url: "https://github.com/karpathy/nanoGPT/blob/master/train.py" },
          { type: "Community Guide", label: "Hugging Face Ultra-Scale Playbook", url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook" }
        ]
      },
      {
        id: "p5-t3",
        codeNum: "5.3",
        animType: "lr_cosine",
        title: "Learning Rate Warmup & Cosine Decay Scheduling",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase5_pretraining",
          fileName: "03_cosine_lr_schedule.py",
          steps: [
            "Create 03_cosine_lr_schedule.py inside phase5_pretraining/ in VS Code.",
            "Paste the warmup + cosine decay learning rate scheduler script below.",
            "Run it to inspect how LR ramps up during warmup and decays smoothly afterward."
          ],
          starterCode: `import math

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

for s in [0, 50, 100, 500, 900, 1000]:
    lr = get_lr(s, warmup_steps=100, max_steps=1000, max_lr=6e-4, min_lr=6e-5)
    print(f"Step {s:4d} -> lr = {lr:.6f}")`,
          runCmd: "python phase5_pretraining/03_cosine_lr_schedule.py",
          expectedOutput: `Step    0 -> lr = 0.000006
Step   50 -> lr = 0.000306
Step  100 -> lr = 0.000600
Step  500 -> lr = 0.000362
Step  900 -> lr = 0.000076
Step 1000 -> lr = 0.000060`
        },
        plainEnglish: "'Linear Warmup' gently ramps up the learning rate over the first few hundred steps, and 'Cosine Decay' smoothly slows down near the finish line so the model settles cleanly into a low-loss valley.",
        liveExample: "Chinchilla, GPT-3, and Llama pretraining schedules all use linear warmup followed by cosine decay down to 10% of peak LR.",
        jargonTerms: [
          { term: "LR Warmup", def: "Gradually increasing learning rate from 0 to peak over early steps." },
          { term: "Cosine Decay", def: "Smoothly lowering learning rate along a gentle wave toward the end of training." }
        ],
        why: "Warming up the learning rate gradually avoids early AdamW variance spikes, while cosine decay settles into sharp minima.",
        lessonSections: [
          {
            heading: "1. Two-Stage Cosine Schedule",
            body: "Linear ramp for step < warmup_steps, followed by 0.5 * (1 + cos(pi * progress)) decay down to min_lr.",
            codeExample: "lr = get_lr(step, warmup_steps=100, max_steps=5000, max_lr=6e-4, min_lr=6e-5)"
          }
        ],
        concepts: [
          "Linear warmup phase (first ~5% of steps).",
          "Cosine annealing decay down to min_lr.",
          "Updating param_group['lr'] dynamically each step."
        ],
        code: `import math\n\ndef get_lr(step, warmup_steps, max_steps, max_lr, min_lr):\n    if step < warmup_steps:\n        return max_lr * (step + 1) / warmup_steps\n    if step >= max_steps:\n        return min_lr\n    progress = (step - warmup_steps) / (max_steps - warmup_steps)\n    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))\n\nprint("Step 0 LR:", get_lr(0, 100, 1000, 6e-4, 6e-5))`,
        quiz: {
          question: "Why do LLM training runs start with a low learning rate that warms up linearly?",
          options: [
            "Because weights are random and AdamW's variance estimates are noisy at step 0, so a huge initial LR causes instability",
            "Because GPUs need time to heat up physically",
            "To skip the first epoch of data"
          ],
          answerIndex: 0,
          explanation: "Warmup stabilizes early attention and LayerNorm statistics before applying peak step sizes."
        },
        pitfall: "Leaving peak learning rate constant for the entire run.",
        project: "Plot your get_lr() schedule curve in matplotlib across 5,000 steps.",
        resourceTitle: "Sebastian Raschka — Appendix D: Learning Rate Warmup & Cosine Decay",
        resourceNote: "Covers learning rate warmup, cosine decay, and gradient clipping in detail.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Appendix D Training Bells & Whistles Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-D" },
          { type: "Community Guide", label: "MachineLearningMastery — Learning Rate Warmup & Cosine Annealing Guide", url: "https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/" }
        ]
      },
      {
        id: "p5-t4",
        codeNum: "5.4",
        animType: "gradient_descent",
        title: "Diagnosing Overfitting, Underfitting & Loss Curves",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase5_pretraining",
          fileName: "04_diagnose_loss_curves.py",
          steps: [
            "Create 04_diagnose_loss_curves.py inside phase5_pretraining/ in VS Code.",
            "Paste the automated loss-curve diagnostic helper below.",
            "Run it to see how ML pipelines detect overfitting vs. healthy convergence."
          ],
          starterCode: `def diagnose_run(train_losses: list[float], val_losses: list[float]) -> str:
    train_end = train_losses[-1]
    val_min = min(val_losses)
    val_end = val_losses[-1]

    if train_end > 2.5 and val_end > 2.5:
        return "UNDERFITTING: Both losses remain high. Increase model size, LR, or training steps."
    if val_end > val_min + 0.15 and train_end < val_end - 0.4:
        return f"OVERFITTING: Validation loss rose from {val_min:.2f} to {val_end:.2f} while train loss fell to {train_end:.2f}."
    return f"HEALTHY CONVERGENCE: Train ({train_end:.2f}) and Val ({val_end:.2f}) improved together."

print("Run A:", diagnose_run([4.2, 2.8, 1.9, 1.6], [4.3, 2.9, 2.0, 1.7]))
print("Run B:", diagnose_run([4.2, 1.8, 0.7, 0.3], [4.3, 2.1, 2.3, 2.8]))`,
          runCmd: "python phase5_pretraining/04_diagnose_loss_curves.py",
          expectedOutput: `Run A: HEALTHY CONVERGENCE: Train (1.60) and Val (1.70) improved together.
Run B: OVERFITTING: Validation loss rose from 2.10 to 2.80 while train loss fell to 0.30.`
        },
        plainEnglish: "Plotting both Training Loss (practice exam) and Validation Loss (unseen final exam) reveals immediately whether your model is memorizing or genuinely learning patterns.",
        liveExample: "ML teams track train/val loss divergence in MLflow or Weights & Biases to trigger early stopping before overfitting wastes GPU hours.",
        jargonTerms: [
          { term: "Overfitting", def: "Train loss drops near 0 while validation loss rises (pure memorization)." },
          { term: "Underfitting", def: "Both train and validation loss stay high (model too small or undertrained)." }
        ],
        why: "Reading train vs. validation loss curves tells you immediately whether your model needs more capacity, more data, or stronger regularization.",
        lessonSections: [
          {
            heading: "1. Reading the Gap Between Train and Validation Loss",
            body: "When train loss continues falling while validation loss bends upward, your model has started memorizing training sequences verbatim.",
            codeExample: "# Healthy: train_loss=1.62, val_loss=1.68\n# Overfit: train_loss=0.41, val_loss=2.45"
          }
        ],
        concepts: [
          "Healthy convergence vs. Overfitting vs. Underfitting.",
          "Logging metrics cleanly to MLflow."
        ],
        code: `# import mlflow\n# mlflow.log_metric("train_loss", train_loss, step=step)\n# mlflow.log_metric("val_loss", val_loss, step=step)\nprint("MLflow logging ready")`,
        quiz: {
          question: "Your training loss is 0.35 (very low), but your validation loss climbed from 1.70 to 2.60. What is happening?",
          options: [
            "Overfitting — the model is memorizing training text instead of generalizing",
            "Underfitting — the model is too small",
            "Perfect convergence"
          ],
          answerIndex: 0,
          explanation: "Diverging train vs. validation loss is the classic signature of overfitting."
        },
        pitfall: "Evaluating only training loss!",
        project: "Log train and validation loss curves with matplotlib/MLflow on your TinyShakespeare run.",
        resourceTitle: "Andrej Karpathy — A Recipe for Training Neural Networks",
        resourceNote: "Timeless practical debugging checklist for diagnosing loss curves and silent bugs.",
        links: [
          { type: "Community Guide", label: "Andrej Karpathy — A Recipe for Training Neural Networks", url: "https://karpathy.github.io/2019/04/25/recipe/" },
          { type: "Blog / Docs", label: "MLflow Official Docs — Tracking PyTorch Experiments", url: "https://mlflow.org/docs/latest/deep-learning/pytorch/index.html" }
        ]
      },
      {
        id: "p5-t5",
        codeNum: "5.5",
        animType: "softmax_logits",
        title: "Autoregressive Sampling Strategies: Greedy, Temperature, Top-k & Top-p (Nucleus)",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase5_pretraining",
          fileName: "05_temperature_top_p_sampling.py",
          steps: [
            "Create 05_temperature_top_p_sampling.py inside phase5_pretraining/ in VS Code.",
            "Paste the Temperature + Top-P Nucleus sampler script below.",
            "Run it to compare deterministic low-temperature probabilities against creative high-temperature probabilities."
          ],
          starterCode: `import torch
import torch.nn.functional as F

tokens = ["Paris", "Lyon", "France", "nice", "banana"]
logits = torch.tensor([[5.2, 3.1, 2.3, 1.4, 0.1]])

def sample_probs(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9):
    scaled = logits / temperature
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = float("-inf")

    filtered_probs = F.softmax(sorted_logits, dim=-1)[0]
    return [(tokens[idx], round(p.item() * 100, 1)) for idx, p in zip(sorted_idx[0], filtered_probs)]

print("Factual  (T=0.3, top_p=0.9):", sample_probs(logits, temperature=0.3, top_p=0.9))
print("Creative (T=1.4, top_p=0.9):", sample_probs(logits, temperature=1.4, top_p=0.9))`,
          runCmd: "python phase5_pretraining/05_temperature_top_p_sampling.py",
          expectedOutput: `Factual  (T=0.3, top_p=0.9): [('Paris', 99.9), ('Lyon', 0.0), ('France', 0.0), ('nice', 0.0), ('banana', 0.0)]
Creative (T=1.4, top_p=0.9): [('Paris', 69.8), ('Lyon', 15.6), ('France', 8.8), ('nice', 5.8), ('banana', 0.0)]`
        },
        plainEnglish: "'Temperature' adjusts creativity by dividing logits before Softmax, while 'Top-k' and 'Top-p' block absurd tail words from ever being chosen.",
        liveExample: "ChatGPT's API exposes exact 'temperature' and 'top_p' parameters backed by this exact Softmax + cumulative mask logic.",
        jargonTerms: [
          { term: "Temperature", def: "Divides logits before Softmax: low (0.2) = focused; high (1.5) = creative." },
          { term: "Top-k / Top-p", def: "Keeps only the top k words or top p% probability mass before sampling." }
        ],
        why: "Greedy decoding repeats loops, while unfiltered random sampling picks bizarre low-probability words. Temperature, Top-k, and Top-p strike the sweet spot.",
        lessonSections: [
          {
            heading: "1. How Temperature Reshapes Logits",
            body: "Dividing logits by T < 1 exaggerates differences between the #1 word and runner-ups; dividing by T > 1 flattens probabilities closer to uniform.",
            codeExample: "probs = F.softmax(logits / temperature, dim=-1)"
          }
        ],
        concepts: [
          "Autoregressive loop cropping context to last max_len tokens.",
          "Temperature scaling (logits / T).",
          "Top-k and Top-p Nucleus sampling."
        ],
        code: `import torch, torch.nn.functional as F\nlogits = torch.tensor([[5.2, 3.1, 2.3, 1.4, 0.1]])\ntemperature, top_p = 0.7, 0.9\nsorted_logits, sorted_idx = torch.sort(logits / temperature, descending=True)\ncum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)\nsorted_mask = cum_probs > top_p\nsorted_mask[..., 1:] = sorted_mask[..., :-1].clone()\nsorted_mask[..., 0] = False\nsorted_logits[sorted_mask] = float('-inf')\nprint("Filtered probs:", F.softmax(sorted_logits, dim=-1))`,
        quiz: {
          question: "What happens when you lower Temperature from 1.0 down to 0.2?",
          options: [
            "The probability distribution sharpens around the highest-logit token, making outputs more deterministic",
            "All vocabulary words become equally likely",
            "The context window doubles"
          ],
          answerIndex: 0,
          explanation: "Dividing logits by a small number (< 1) amplifies the gap between the top logit and alternatives."
        },
        pitfall: "Forgetting to slice context to the last max_len tokens (idx[:, -cfg.context_length:]) inside the generation loop!",
        project: "Implement a unified generate(idx, max_new_tokens, temperature, top_k, top_p) method.",
        resourceTitle: "Hugging Face Blog — How to Generate Text (Top-k & Top-p)",
        resourceNote: "Visualizes probability bar charts before and after Temperature scaling and Top-k filtering.",
        links: [
          { type: "Community Guide", label: "Hugging Face Blog — How to Generate Text: Greedy, Beam, Top-k & Top-p", url: "https://huggingface.co/blog/how-to-generate" },
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5 Decoding Strategies Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" }
        ]
      }
    ]
  },
  {
    id: "phase-6",
    number: "06",
    category: "training",
    accent: "teal",
    duration: "Week 11–12",
    title: "Fine-Tuning, Alignment & Local Deployment",
    subtitle: "Load open pretrained weights into your own PyTorch code, fine-tune for chat instructions with LoRA, and deploy behind FastAPI + Ollama.",
    plainBigPicture: "A base pretrained model is only an autocomplete engine—not yet a helpful Q&A assistant. In this final phase, you load open pretrained weights into your own code, teach the model to answer questions using 'SFT' and 'LoRA', and wrap it in a live web API.",
    liveAiBridge: {
      headline: "Where Live Running AIs Use Phase 6 Skills",
      summary: "Every custom enterprise AI assistant and local coding copilot runs this exact post-training & serving stack:",
      examples: [
        "Multi-tenant cloud platforms serve hundreds of custom fine-tuned models from a single base GPU by hot-swapping tiny 20MB LoRA adapter matrices (Module 6.3) per request.",
        "Ollama and llama.cpp serve 4-bit quantized GGUF checkpoints locally on consumer laptops (Modules 6.3 & 6.5)."
      ]
    },
    jargonDecoder: [
      { term: "SFT", def: "Fine-tuning a base model on Q&A chat pairs" },
      { term: "LoRA", def: "Training tiny adapter matrices instead of all weights" },
      { term: "Quantization", def: "Compressing weights (32-bit -> 4-bit) to fit laptop RAM" },
      { term: "DPO / RLHF", def: "Aligning replies to human helpfulness preferences" },
      { term: "FastAPI / Ollama", def: "Serving your model locally as a REST API" }
    ],
    achievements: [
      "Map open pretrained weights (such as GPT-2 124M) directly into your from-scratch PyTorch classes and verify exact logit parity.",
      "Fine-tune a base LLM into a Q&A assistant using Supervised Fine-Tuning (SFT) and memory-saving LoRA adapters.",
      "Deploy your custom model locally behind a FastAPI + Uvicorn REST endpoint and benchmark it alongside Ollama."
    ],
    miniProject: "Fine-Tune a Small Model on Custom Q&A Data & Serve Locally via FastAPI + Uvicorn",
    topics: [
      {
        id: "p6-t1",
        codeNum: "6.1",
        animType: "transformer_block",
        title: "Loading Open Pretrained Weights (GPT-2 / Llama) into Custom PyTorch Code",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase6_finetune_serve",
          fileName: "01_inspect_gpt2_weights.py",
          steps: [
            "Create a new folder phase6_finetune_serve/ inside VIT-Automation in VS Code.",
            "Create 01_inspect_gpt2_weights.py and paste the state_dict weight-inspection script below.",
            "Run it to inspect real pretrained GPT-2 layer names and tensor shapes."
          ],
          starterCode: `from transformers import AutoModelForCausalLM

# Download & inspect real pretrained GPT-2 Small (124M) state_dict keys
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
sd = hf_model.state_dict()

print(f"Total tensors in GPT-2 checkpoint: {len(sd)}")
for key in list(sd.keys())[:8]:
    print(f"{key:38s} | shape = {tuple(sd[key].shape)}")`,
          runCmd: "python phase6_finetune_serve/01_inspect_gpt2_weights.py",
          expectedOutput: `Total tensors in GPT-2 checkpoint: 149
transformer.wte.weight                 | shape = (50257, 768)
transformer.wpe.weight                 | shape = (1024, 768)
transformer.h.0.ln_1.weight            | shape = (768,)
transformer.h.0.ln_1.bias              | shape = (768,)
transformer.h.0.attn.c_attn.weight     | shape = (768, 2304)`
        },
        plainEnglish: "Copy free open pretrained checkpoint numbers into your own from-scratch PyTorch classes—proving your architecture is 100% authentic.",
        liveExample: "Inference engines like vLLM and llama.cpp map Hugging Face state_dict tensors into custom high-speed model classes using this exact key-matching process.",
        jargonTerms: [
          { term: "state_dict()", def: "Python dictionary mapping every layer name to its saved weight tensor." }
        ],
        why: "Pretraining from scratch on web-scale data costs millions, so real-world builders map pretrained checkpoint tensors directly into their own PyTorch architecture.",
        lessonSections: [
          {
            heading: "1. Verifying Tensor Shapes Before Copying",
            body: "Always assert source_tensor.shape == target_tensor.shape when mapping external weights into your custom GPTModel.",
            codeExample: "assert target.shape == source.shape, f'Mismatch: {target.shape} vs {source.shape}'"
          }
        ],
        concepts: [
          "Inspecting state_dict() keys and tensor shapes.",
          "Handling Conv1D vs. nn.Linear weight transpositions.",
          "Verifying exact logit parity."
        ],
        code: `# for k, v in list(model.state_dict().items())[:6]:\n#     print(f"{k:35s} | {tuple(v.shape)}")\nprint("Ready to map pretrained checkpoint tensors")`,
        quiz: {
          question: "Why do OpenAI's original GPT-2 linear weights need .T (transposition) when loaded into PyTorch nn.Linear?",
          options: [
            "OpenAI stored Conv1D weights as (in_features, out_features), whereas PyTorch nn.Linear stores (out_features, in_features)",
            "To convert integers to strings",
            "To reduce parameter count by 50%"
          ],
          answerIndex: 0,
          explanation: "Checking .shape catches transposed weight layouts immediately before copying tensors."
        },
        pitfall: "OpenAI's original GPT-2 checkpoints store linear weights transposed compared to PyTorch's nn.Linear!",
        project: "Load pretrained GPT-2 Small (124M) weights into your from-scratch PyTorch GPTModel.",
        resourceTitle: "Sebastian Raschka — Loading OpenAI Pretrained Weights & nanoGPT",
        resourceNote: "Step-by-step weight mapping function from pretrained tensors into custom GPTModel layers.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 5.5 Loading GPT-2 Weights Notebook", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05" },
          { type: "Community Guide", label: "Andrej Karpathy — nanoGPT from_pretrained GPT-2 Implementation", url: "https://github.com/karpathy/nanoGPT/blob/master/model.py" }
        ]
      },
      {
        id: "p6-t2",
        codeNum: "6.2",
        animType: "sliding_window",
        title: "Supervised Fine-Tuning (SFT) & Instruction Dataset Formatting",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase6_finetune_serve",
          fileName: "02_sft_prompt_masking.py",
          steps: [
            "Create 02_sft_prompt_masking.py inside phase6_finetune_serve/ in VS Code.",
            "Paste the SFT instruction formatter & ignore_index=-100 target masking script.",
            "Run it to verify how prompt tokens are masked so loss grades only the assistant's reply."
          ],
          starterCode: `import torch
import torch.nn.functional as F

# Simulate token IDs: [USER_PROMPT (3 tokens)] + [ASSISTANT_REPLY (3 tokens)] + [PAD (2 tokens)]
prompt_ids = [101, 2054, 2003]
reply_ids  = [1037, 6898, 50256]
pad_ids    = [0, 0]

input_ids = torch.tensor([prompt_ids + reply_ids + pad_ids])
# Mask prompt and padding tokens with -100 so F.cross_entropy only grades the reply!
target_ids = torch.tensor([[-100, -100, -100] + reply_ids + [-100, -100]])

vocab_size = 50257
dummy_logits = torch.randn(1, input_ids.size(1), vocab_size)
sft_loss = F.cross_entropy(dummy_logits.view(-1, vocab_size), target_ids.view(-1), ignore_index=-100)

print("Input IDs: ", input_ids[0].tolist())
print("Target IDs:", target_ids[0].tolist())
print(f"Masked SFT Loss: {sft_loss.item():.4f}")`,
          runCmd: "python phase6_finetune_serve/02_sft_prompt_masking.py",
          expectedOutput: `Input IDs:  [101, 2054, 2003, 1037, 6898, 50256, 0, 0]
Target IDs: [-100, -100, -100, 1037, 6898, 50256, -100, -100]
Masked SFT Loss: 11.1492`
        },
        plainEnglish: "'Supervised Fine-Tuning (SFT)' trains a base autocomplete model on curated Instruction -> Answer pairs so it behaves like a helpful assistant.",
        liveExample: "Meta's Llama-3-Instruct and OpenAI's GPT-3.5-Turbo were transformed from document completers into chat assistants via SFT.",
        jargonTerms: [
          { term: "SFT", def: "Supervised Fine-Tuning on formatted (Instruction, Response) examples." },
          { term: "ignore_index=-100", def: "Tells PyTorch's loss function to skip padding tokens when grading." }
        ],
        why: "SFT teaches a base model to follow instructions and answer as an assistant.",
        lessonSections: [
          {
            heading: "1. Masking Padding & Prompt Tokens with -100",
            body: "Setting target IDs to -100 on padding tokens instructs F.cross_entropy to ignore them so loss focuses purely on assistant responses.",
            codeExample: "loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)"
          }
        ],
        concepts: [
          "Structuring instruction-input-response JSON records into prompt templates.",
          "Target padding with ignore_index = -100.",
          "Using a low learning rate (2e-5) for 2–3 epochs."
        ],
        code: `import torch.nn.functional as F\n# loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)\nprint("SFT loss masking configured with ignore_index=-100")`,
        quiz: {
          question: "Why do SFT collators fill padding positions in the target tensor with -100?",
          options: [
            "PyTorch's F.cross_entropy has ignore_index=-100 by default, skipping padded slots in loss computation",
            "Because -100 is the token ID for space",
            "To double the learning rate on padding"
          ],
          answerIndex: 0,
          explanation: "Ignoring padding tokens prevents the model from wasting capacity predicting meaningless pad tokens."
        },
        pitfall: "Training SFT for 20+ epochs on a tiny dataset causes catastrophic forgetting.",
        project: "Curate a 100-item IT/Python Q&A dataset and fine-tune your model to answer domain questions.",
        resourceTitle: "Sebastian Raschka — Chapter 7: Instruction Fine-Tuning & Hugging Face SFT",
        resourceNote: "Covers custom collate functions, -100 padding masks, and instruction tuning end to end.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Chapter 7 Instruction Fine-Tuning Code", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07" },
          { type: "Blog / Docs", label: "Hugging Face LLM Course — Chapter 3: Fine-Tuning a Pretrained Model", url: "https://huggingface.co/learn/llm-course/chapter3/1" }
        ]
      },
      {
        id: "p6-t3",
        codeNum: "6.3",
        animType: "lora_lowrank",
        title: "Parameter-Efficient Fine-Tuning (LoRA) & Quantization Basics",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase6_finetune_serve",
          fileName: "03_lora_adapter_pytorch.py",
          steps: [
            "Create 03_lora_adapter_pytorch.py inside phase6_finetune_serve/ in VS Code.",
            "Paste the pure-PyTorch LinearWithLoRA wrapper below.",
            "Run it to verify that trainable parameters drop by >95% and step-0 output matches the frozen base layer."
          ],
          starterCode: `import torch
import torch.nn as nn

class LinearWithLoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False  # Freeze original pretrained weights

        in_dim, out_dim = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A @ self.B) * self.scale

if __name__ == "__main__":
    base = nn.Linear(512, 512, bias=False)
    lora_layer = LinearWithLoRA(base, rank=8)

    total = sum(p.numel() for p in lora_layer.parameters())
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    x = torch.randn(1, 4, 512)
    diff = (base(x) - lora_layer(x)).abs().max().item()

    print(f"Total params: {total:,} | Trainable LoRA params: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Step-0 output difference vs base model: {diff:.8f}")`,
          runCmd: "python phase6_finetune_serve/03_lora_adapter_pytorch.py",
          expectedOutput: `Total params: 270,336 | Trainable LoRA params: 8,192 (3.03%)
Step-0 output difference vs base model: 0.00000000`
        },
        plainEnglish: "'LoRA' freezes the original brain and attaches tiny side-matrices (A and B) that learn only the new task—cutting memory by 90%! 'Quantization' compresses weights to 4-bit integers.",
        liveExample: "Apple Intelligence on iPhone/Mac runs a ~3B parameter on-device LLM using 2-bit/4-bit quantization and task-specific LoRA adapters.",
        jargonTerms: [
          { term: "LoRA", def: "Low-Rank Adaptation—training tiny side matrices (A @ B) while base weights stay frozen." },
          { term: "Quantization", def: "Compressing model weights from 32-bit floats to 8-bit or 4-bit numbers." }
        ],
        why: "LoRA freezes base weights and trains tiny low-rank adapter matrices (~1% of parameters), while quantization shrinks RAM 4x.",
        lessonSections: [
          {
            heading: "1. Zero-Initializing LoRA Matrix B",
            body: "Initializing B = 0 ensures ΔW = A @ B = 0 at step 0, so fine-tuning begins from the exact pretrained model without random noise.",
            codeExample: "self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)\nself.B = nn.Parameter(torch.zeros(rank, out_dim))"
          }
        ],
        concepts: [
          "LoRA math: freezing W and learning ΔW = A @ B with rank r = 8 or 16.",
          "Initializing B = 0 and A ~ N(0, sigma^2).",
          "Quantization: FP32 -> BF16 -> INT8 -> INT4/GGUF."
        ],
        code: `import torch, torch.nn as nn\n\nclass LoRALayer(nn.Module):\n    def __init__(self, in_dim, out_dim, rank=8, alpha=16):\n        super().__init__()\n        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)\n        self.B = nn.Parameter(torch.zeros(rank, out_dim))\n        self.scale = alpha / rank\n    def forward(self, x):\n        return (x @ self.A @ self.B) * self.scale`,
        quiz: {
          question: "Why is LoRA matrix B initialized to all zeros while matrix A is initialized with small random values?",
          options: [
            "So ΔW = A @ B equals 0 at step 0, preserving the exact pretrained model behavior at the start of training",
            "Because zero matrices multiply faster",
            "To freeze matrix A"
          ],
          answerIndex: 0,
          explanation: "If both A and B were random, step 0 would inject random noise into every adapted layer."
        },
        pitfall: "Forgetting to freeze base model parameters (param.requires_grad = False) before training LoRA!",
        project: "Implement a LinearWithLoRA wrapper in pure PyTorch and verify trainable parameter count drops by >95%.",
        resourceTitle: "Sebastian Raschka — LoRA From Scratch in PyTorch (Appendix E)",
        resourceNote: "Implements LoRA from scratch in pure PyTorch with clear rank/alpha explanations.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — Appendix E: LoRA From Scratch in PyTorch", url: "https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E" },
          { type: "Community Guide", label: "Maarten Grootendorst — A Visual Guide to Quantization", url: "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization" }
        ]
      },
      {
        id: "p6-t4",
        codeNum: "6.4",
        animType: "softmax_logits",
        title: "Evaluation Benchmarks, Hallucination Checks & Alignment (DPO Intuition)",
        duration: "2 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase6_finetune_serve",
          fileName: "04_eval_rubric_test.py",
          steps: [
            "Create 04_eval_rubric_test.py inside phase6_finetune_serve/ in VS Code.",
            "Paste the automated evaluation rubric & DPO preference check script below.",
            "Run it with python or pytest to verify deterministic grading."
          ],
          starterCode: `EVAL_CASES = [
    {"prompt": "What port does HTTPS use?", "expected_keyword": "443", "response": "HTTPS uses TCP port 443."},
    {"prompt": "What does DNS stand for?", "expected_keyword": "Domain Name System", "response": "DNS stands for Domain Name System."},
]

def run_eval_suite(cases):
    passed = 0
    for item in cases:
        ok = item["expected_keyword"].lower() in item["response"].lower()
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {item['prompt']} -> {item['response']}")
    print(f"Benchmark Score: {passed}/{len(cases)} ({passed/len(cases)*100:.0f}%)")

if __name__ == "__main__":
    run_eval_suite(EVAL_CASES)`,
          runCmd: "python phase6_finetune_serve/04_eval_rubric_test.py",
          expectedOutput: `[PASS] What port does HTTPS use? -> HTTPS uses TCP port 443.
[PASS] What does DNS stand for? -> DNS stands for Domain Name System.
Benchmark Score: 2/2 (100%)`
        },
        plainEnglish: "Before shipping a model, we run automated 'pytest' checks on unseen prompts with temperature=0, and use preference alignment ('DPO') to reward honest, helpful replies.",
        liveExample: "Zephyr, Llama-3-Instruct, and Qwen-Chat all apply Direct Preference Optimization (DPO) after SFT to reduce hallucinations and improve helpfulness.",
        jargonTerms: [
          { term: "Hallucination", def: "When an LLM confidently generates false facts not supported by context." },
          { term: "DPO / RLHF", def: "Training the model to prefer human-approved answers over rejected ones." }
        ],
        why: "Evaluating quality requires automated benchmarks, deterministic rubric checks, and preference alignment (RLHF / DPO).",
        lessonSections: [
          {
            heading: "1. Deterministic Evaluation with temperature=0.0",
            body: "Always set model.eval() and greedy decoding (temperature=0.0) when running automated pytest regressions so test results are 100% reproducible.",
            codeExample: "@torch.no_grad()\ndef evaluate_prompt(model, ids):\n    model.eval()"
          }
        ],
        concepts: [
          "Automated evaluation: held-out task accuracy and schema validation.",
          "Direct Preference Optimization (DPO) on (chosen, rejected) pairs.",
          "Deterministic inference (temperature=0.0)."
        ],
        code: `import torch\n@torch.no_grad()\ndef evaluate_prompt(model, prompt_ids):\n    model.eval()\n    return prompt_ids`,
        quiz: {
          question: "What training data format does Direct Preference Optimization (DPO) require?",
          options: [
            "Triplets of (Prompt, Chosen Good Response, Rejected Bad Response)",
            "Unlabeled raw text only",
            "Image-to-audio pairs"
          ],
          answerIndex: 0,
          explanation: "DPO directly increases the log-probability margin between preferred (chosen) and dispreferred (rejected) completions."
        },
        pitfall: "Evaluating your model on prompts that accidentally leaked into your SFT training split!",
        project: "Build a 20-prompt automated evaluation script using pytest.",
        resourceTitle: "Sebastian Raschka — DPO Alignment & LLM Evaluation",
        resourceNote: "Demonstrates automated scoring of fine-tuned instruction responses.",
        links: [
          { type: "Community Guide", label: "Sebastian Raschka — LLM Alignment: RLHF vs. DPO Explained", url: "https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives" },
          { type: "Blog / Docs", label: "Hugging Face TRL Docs — Direct Preference Optimization (DPO)", url: "https://huggingface.co/docs/trl/main/en/dpo_trainer" }
        ]
      },
      {
        id: "p6-t5",
        codeNum: "6.5",
        animType: "serving_api",
        title: "Serving Your Model Locally with FastAPI, Uvicorn & Ollama",
        duration: "3 Days",
        vsCodeLab: {
          folder: "VIT-Automation/phase6_finetune_serve",
          fileName: "05_serve_llm_fastapi.py",
          steps: [
            "Create 05_serve_llm_fastapi.py inside phase6_finetune_serve/ in VS Code.",
            "Paste the FastAPI + Pydantic REST endpoint code below and save (Ctrl + S).",
            "Start the server with uvicorn phase6_finetune_serve.05_serve_llm_fastapi:app --reload and open http://127.0.0.1:8000/docs in your browser!"
          ],
          starterCode: `from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI(title="From-Scratch LLM Local API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 40
    temperature: float = 0.7

@app.get("/health")
def health_check():
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}

@app.post("/v1/generate")
def generate_text(req: GenerateRequest):
    with torch.no_grad():
        completion = f"{req.prompt} -> [Generated {req.max_tokens} tokens at T={req.temperature}]"
    return {"model": "mini-gpt-from-scratch", "output": completion}`,
          runCmd: "uvicorn phase6_finetune_serve.05_serve_llm_fastapi:app --reload",
          expectedOutput: `INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.`
        },
        plainEnglish: "'FastAPI' creates a clean HTTP endpoint (/v1/generate) where any app can send a prompt and get JSON back, 'Uvicorn' runs that web server locally, and 'Ollama' runs local GGUF models.",
        liveExample: "Ollama exposes an OpenAI-compatible localhost REST API on port 11434 using this exact JSON request/response contract.",
        jargonTerms: [
          { term: "FastAPI", def: "Modern Python framework for building fast, auto-documented REST APIs." },
          { term: "Uvicorn", def: "Lightweight web server that runs your FastAPI app on localhost." }
        ],
        why: "Completing the engineering loop: package your PyTorch model behind a typed REST API using FastAPI + Uvicorn.",
        lessonSections: [
          {
            heading: "1. Load Weights Once at Startup",
            body: "Instantiate your GPTModel and load its checkpoint once at module startup so every POST /v1/generate request runs in milliseconds.",
            codeExample: "@app.post('/v1/generate')\ndef generate_endpoint(req: GenerateRequest):\n    with torch.no_grad():\n        return {'output': 'generated text'}"
          }
        ],
        concepts: [
          "Loading model weights once at FastAPI startup.",
          "Validating request/response JSON payloads with Pydantic BaseModel.",
          "Running local quantized models via Ollama."
        ],
        code: `from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI(title="My From-Scratch LLM API")\n\nclass GenerateRequest(BaseModel):\n    prompt: str\n    max_tokens: int = 60\n    temperature: float = 0.7`,
        quiz: {
          question: "Where should you load your PyTorch model checkpoint inside a FastAPI application?",
          options: [
            "Once at application startup outside the route handler, so requests don't reload gigabytes from disk",
            "Inside the @app.post('/generate') function on every request",
            "Inside the Pydantic BaseModel class"
          ],
          answerIndex: 0,
          explanation: "Loading weights once into GPU/CPU memory at startup keeps per-request inference latency down to milliseconds."
        },
        pitfall: "Reloading the PyTorch checkpoint file inside the /generate route handler on every request!",
        project: "Launch your custom LLM server with 'uvicorn app:app --reload' and test it via Swagger UI (/docs).",
        resourceTitle: "FastAPI Official Docs & Ollama Quickstart",
        resourceNote: "Brings together PyTorch, FastAPI, Uvicorn, requests, and Ollama from your environment.",
        links: [
          { type: "Community Guide", label: "Real Python — Using FastAPI to Build Python Web APIs", url: "https://realpython.com/fastapi-python-web-apis/" },
          { type: "Blog / Docs", label: "FastAPI Official Tutorial — First Steps & Automatic /docs UI", url: "https://fastapi.tiangolo.com/tutorial/first-steps/" },
          { type: "Blog / Docs", label: "Ollama Official Quickstart & REST API Documentation", url: "https://docs.ollama.com/quickstart" }
        ]
      }
    ]
  }
];

const ALL_TOPICS = ROADMAP_PHASES.flatMap((phase) =>
  phase.topics.map((t) => ({
    ...t,
    phaseId: phase.id,
    phaseNumber: phase.number,
    phaseTitle: phase.title
  }))
);

// =========================================================================
// 2. INTERACTIVE ANIMATED CONCEPT VISUALIZERS (ENHANCED ZERO-TO-HERO LABS)
// =========================================================================
const BPE_STEP_SCENARIOS = [
  {
    step: 0,
    title: "Raw Input String",
    desc: "Before an LLM can process text, raw Unicode strings are ingested as a single unsegmented sequence.",
    mergeRule: "None (Raw UTF-8 Text)",
    tokens: [{ id: "raw-0", text: "unbelievable", tokenId: "—", type: "raw", bytes: 12 }]
  },
  {
    step: 1,
    title: "Character / UTF-8 Byte Split",
    desc: "Byte-Pair Encoding (BPE) starts by splitting text into base atomic characters or UTF-8 bytes (0–255).",
    mergeRule: "Base Vocabulary Split (256 byte tokens)",
    tokens: "unbelievable".split("").map((char, idx) => ({
      id: `char-${idx}`,
      text: char,
      tokenId: char.charCodeAt(0),
      type: "char",
      bytes: 1
    }))
  },
  {
    step: 2,
    title: 'Frequent Pair Merge #1 ("un" & "le")',
    desc: "The tokenizer scans corpus statistics and merges the most frequent adjacent character pairs.",
    mergeRule: "Merge ('u', 'n') → 'un'  |  Merge ('l', 'e') → 'le'",
    tokens: [
      { id: "m-un", text: "un", tokenId: 403, type: "merged", bytes: 2 },
      { id: "char-2", text: "b", tokenId: 98, type: "char", bytes: 1 },
      { id: "char-3", text: "e", tokenId: 101, type: "char", bytes: 1 },
      { id: "char-4", text: "l", tokenId: 108, type: "char", bytes: 1 },
      { id: "char-5", text: "i", tokenId: 105, type: "char", bytes: 1 },
      { id: "char-6", text: "e", tokenId: 101, type: "char", bytes: 1 },
      { id: "char-7", text: "v", tokenId: 118, type: "char", bytes: 1 },
      { id: "char-8", text: "a", tokenId: 97, type: "char", bytes: 1 },
      { id: "char-9", text: "b", tokenId: 98, type: "char", bytes: 1 },
      { id: "m-le", text: "le", tokenId: 294, type: "merged", bytes: 2 }
    ]
  },
  {
    step: 3,
    title: 'Morphological Suffix Merge ("able")',
    desc: "Subword roots and common English suffixes merge into single high-density semantic tokens.",
    mergeRule: "Merge ('a', 'b', 'le') → 'able'",
    tokens: [
      { id: "m-un", text: "un", tokenId: 403, type: "merged", bytes: 2 },
      { id: "char-2", text: "b", tokenId: 98, type: "char", bytes: 1 },
      { id: "char-3", text: "e", tokenId: 101, type: "char", bytes: 1 },
      { id: "char-4", text: "l", tokenId: 108, type: "char", bytes: 1 },
      { id: "char-5", text: "i", tokenId: 105, type: "char", bytes: 1 },
      { id: "char-6", text: "e", tokenId: 101, type: "char", bytes: 1 },
      { id: "char-7", text: "v", tokenId: 118, type: "char", bytes: 1 },
      { id: "m-able", text: "able", tokenId: 540, type: "merged", bytes: 4 }
    ]
  },
  {
    step: 4,
    title: "Final Subword Vocabulary IDs",
    desc: "The 12-character word is compressed into 3 dense integer IDs ready for Embedding Matrix lookup.",
    mergeRule: "Merge ('b','e','l','i','e','v') → 'believ' (3 tokens total)",
    tokens: [
      { id: "m-un", text: "un", tokenId: 403, type: "final-prefix", bytes: 2 },
      { id: "m-believ", text: "believ", tokenId: 18924, type: "final-stem", bytes: 6 },
      { id: "m-able", text: "able", tokenId: 540, type: "final-suffix", bytes: 4 }
    ]
  }
];

function MotionBPELab() {
  const [mode, setMode] = useState("stepper");
  const [stepIdx, setStepIdx] = useState(0);
  const [customText, setCustomText] = useState("unbelievable tokenization");
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

  const liveCustomTokens = useMemo(() => {
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
        for (let c = 0; c < tok.length; c++) id = ((id * 31 + tok.charCodeAt(c)) % 49000) + 256;
      }
      return { id: `c-${i}-${tok}`, text: tok, tokenId: id, bytes: new TextEncoder().encode(tok).length };
    });
  }, [customText]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md">
              Interactive Lab • motion/react
            </span>
            <h3 className="text-base font-extrabold text-[#17394D]">Byte-Pair Encoding (BPE) Tokenization Stepper</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Step through how 12 raw UTF-8 characters merge into 3 dense subword IDs—or test your own custom sentence.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-lg bg-slate-200/70 p-0.5 text-xs font-semibold">
            <button
              type="button"
              onClick={() => setMode("stepper")}
              className={`px-2.5 py-1 rounded-md transition ${
                mode === "stepper" ? "bg-white text-[#17394D] shadow-2xs" : "text-slate-600"
              }`}
            >
              5-Step Merge Walkthrough
            </button>
            <button
              type="button"
              onClick={() => setMode("sandbox")}
              className={`px-2.5 py-1 rounded-md transition ${
                mode === "sandbox" ? "bg-white text-[#17394D] shadow-2xs" : "text-slate-600"
              }`}
            >
              Live Text Sandbox
            </button>
          </div>
        </div>
      </div>

      {mode === "stepper" ? (
        <>
          <div className="grid grid-cols-5 gap-1.5 px-6 pt-4">
            {BPE_STEP_SCENARIOS.map((s, i) => (
              <button
                key={s.step}
                type="button"
                onClick={() => setStepIdx(i)}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i <= stepIdx ? "bg-indigo-600" : "bg-slate-200 hover:bg-slate-300"
                }`}
                title={`Step ${i}: ${s.title}`}
              />
            ))}
          </div>

          <div className="p-6 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="text-xs font-bold text-indigo-600">
                  Step {current.step} of {BPE_STEP_SCENARIOS.length - 1} — {current.title}
                </span>
                <p className="text-xs sm:text-sm text-slate-600 mt-0.5">{current.desc}</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setStepIdx(0)}
                  disabled={stepIdx === 0}
                  className="px-2.5 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-100 disabled:opacity-40"
                >
                  Reset
                </button>
                <button
                  type="button"
                  onClick={() => setStepIdx((s) => Math.max(0, s - 1))}
                  disabled={stepIdx === 0}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                >
                  &larr; Prev
                </button>
                <button
                  type="button"
                  onClick={() => setStepIdx((s) => Math.min(BPE_STEP_SCENARIOS.length - 1, s + 1))}
                  disabled={stepIdx === BPE_STEP_SCENARIOS.length - 1}
                  className="px-3.5 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 shadow-2xs"
                >
                  Next Merge &rarr;
                </button>
              </div>
            </div>

            <div className="min-h-[116px] p-5 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-wrap items-center justify-center gap-2.5">
              {current.tokens.map((tok) => (
                <div
                  key={tok.id}
                  className={`px-3.5 py-2 rounded-xl border-2 font-mono flex flex-col items-center transition-all duration-300 hover:-translate-y-0.5 ${getTokenStyle(
                    tok.type
                  )}`}
                >
                  <span className="text-base font-bold tracking-tight">&quot;{tok.text}&quot;</span>
                  <div className="flex items-center gap-1.5 mt-1 text-[10px] opacity-80">
                    <span>ID: {tok.tokenId}</span>
                    <span>&bull;</span>
                    <span>{tok.bytes}B</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 text-xs">
              <div className="flex items-center gap-2 text-slate-600">
                <span className="font-semibold">Active Rule:</span>
                <code className="font-mono font-semibold text-slate-800 bg-slate-100 px-2 py-0.5 rounded">
                  {current.mergeRule}
                </code>
              </div>
              <div className="text-slate-500 font-mono">
                Tensor Input:{" "}
                <span className="text-indigo-600 font-bold">
                  [{current.tokens.map((t) => t.tokenId).join(", ")}]
                </span>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="p-6 space-y-4">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="Type any phrase to tokenize..."
              className="flex-1 px-3.5 py-2 rounded-xl border border-slate-300 text-xs sm:text-sm font-mono text-[#17394D]"
            />
            <div className="px-3 py-2 rounded-xl bg-slate-50 border border-slate-200 font-mono text-xs text-slate-600 flex items-center gap-2">
              <span>Compression:</span>
              <strong className="text-indigo-700">
                {customText.length} chars &rarr; {liveCustomTokens.length} tokens
              </strong>
            </div>
          </div>
          <div className="min-h-[100px] p-4 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-wrap items-center justify-center gap-2">
            {liveCustomTokens.map((tok) => (
              <div
                key={tok.id}
                className={`px-3 py-1.5 rounded-xl border-2 font-mono flex flex-col items-center ${
                  tok.text.length > 1
                    ? "bg-emerald-50 border-emerald-300 text-emerald-900"
                    : "bg-sky-50 border-sky-200 text-sky-900"
                }`}
              >
                <span className="text-sm font-bold">&quot;{tok.text === " " ? "␣" : tok.text}&quot;</span>
                <span className="text-[10px] opacity-80">
                  ID:{tok.tokenId} &bull; {tok.bytes}B
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const EMBEDDING_3D_NODES = [
  { id: "King", word: "King", x: 1.35, y: 1.25, z: 0.45, cluster: "Royalty", color: "#4f46e5", vec: "[0.82, 0.76, 0.28]" },
  { id: "Queen", word: "Queen", x: 1.65, y: 1.45, z: -0.35, cluster: "Royalty", color: "#4f46e5", vec: "[0.91, 0.85, -0.21]" },
  { id: "PyTorch", word: "PyTorch", x: -1.45, y: -0.95, z: 0.85, cluster: "Deep Learning", color: "#059669", vec: "[-0.88, -0.58, 0.52]" },
  { id: "Tensor", word: "Tensor", x: -1.25, y: -1.35, z: 0.55, cluster: "Deep Learning", color: "#059669", vec: "[-0.76, -0.82, 0.34]" },
  { id: "Transformer", word: "Transformer", x: -0.25, y: 1.15, z: -1.25, cluster: "Architecture", color: "#d97706", vec: "[-0.15, 0.70, -0.76]" },
  { id: "Attention", word: "Attention", x: 0.15, y: 0.85, z: -1.55, cluster: "Architecture", color: "#d97706", vec: "[0.09, 0.52, -0.94]" }
];

const EMBEDDING_SIMILARITY_LINKS = [
  { from: "King", to: "Queen", score: "0.94", color: "#6366f1" },
  { from: "PyTorch", to: "Tensor", score: "0.91", color: "#10b981" },
  { from: "Transformer", to: "Attention", score: "0.96", color: "#f59e0b" }
];

function ThreeDEmbeddingLab() {
  const [rotY, setRotY] = useState(0.55);
  const [rotX, setRotX] = useState(0.35);
  const [isDragging, setIsDragging] = useState(false);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [autoSpin, setAutoSpin] = useState(true);
  const [showSinusoid, setShowSinusoid] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!autoSpin || isDragging) return;
    const interval = setInterval(() => setRotY((r) => r + 0.008), 30);
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

  const projectedNodes = useMemo(() => {
    return EMBEDDING_3D_NODES.map((node, idx) => {
      const pxOffset = showSinusoid ? Math.cos(idx * 1.1) * 0.22 : 0;
      const pyOffset = showSinusoid ? Math.sin(idx * 1.1) * 0.22 : 0;
      return {
        ...node,
        proj: projectPoint(node.x + pxOffset, node.y + pyOffset, node.z)
      };
    }).sort((a, b) => b.proj.depth - a.proj.depth);
  }, [rotX, rotY, showSinusoid]);

  const nodeMap = useMemo(() => {
    const map = {};
    projectedNodes.forEach((n) => (map[n.id] = n));
    return map;
  }, [projectedNodes]);

  const origin = projectPoint(0, 0, 0);
  const xAxisPos = projectPoint(2.2, 0, 0);
  const xAxisNeg = projectPoint(-2.2, 0, 0);
  const yAxisPos = projectPoint(0, 2.0, 0);
  const yAxisNeg = projectPoint(0, -2.0, 0);
  const zAxisPos = projectPoint(0, 0, 2.2);
  const zAxisNeg = projectPoint(0, 0, -2.2);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-700 rounded-md">
              Interactive Lab • @react-three/fiber
            </span>
            <h3 className="text-base font-extrabold text-[#17394D]">3D Semantic Embedding Space (PCA Projection)</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Drag to orbit 3D vector space, hover nodes to inspect coordinates, and toggle Positional Encoding shifts.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowSinusoid((s) => !s)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${
              showSinusoid
                ? "bg-teal-50 border-teal-300 text-teal-800"
                : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            Positional Offset: {showSinusoid ? "ON" : "OFF"}
          </button>
          <button
            type="button"
            onClick={() => setAutoSpin((s) => !s)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${
              autoSpin
                ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {autoSpin ? "Auto-Orbit: ON" : "Auto-Orbit: OFF"}
          </button>
          <button
            type="button"
            onClick={() => {
              setRotX(0.35);
              setRotY(0.55);
            }}
            className="px-2.5 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="p-5 space-y-3">
        <div
          onPointerDown={(e) => {
            setIsDragging(true);
            setAutoSpin(false);
            dragStart.current = { x: e.clientX, y: e.clientY };
          }}
          onPointerMove={(e) => {
            if (!isDragging) return;
            const dx = e.clientX - dragStart.current.x;
            const dy = e.clientY - dragStart.current.y;
            dragStart.current = { x: e.clientX, y: e.clientY };
            setRotY((r) => r + dx * 0.01);
            setRotX((r) => Math.max(-1.2, Math.min(1.2, r + dy * 0.01)));
          }}
          onPointerUp={() => setIsDragging(false)}
          onPointerLeave={() => setIsDragging(false)}
          className="relative w-full h-[310px] rounded-xl bg-slate-50 border border-slate-200 overflow-hidden cursor-grab active:cursor-grabbing select-none"
        >
          <svg viewBox="0 0 520 310" className="w-full h-full">
            <defs>
              <radialGradient id="embedBgGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ffffff" />
                <stop offset="100%" stopColor="#f1f5f9" />
              </radialGradient>
            </defs>
            <rect width="520" height="310" fill="url(#embedBgGlow)" />

            <line x1={xAxisNeg.px} y1={xAxisNeg.py} x2={xAxisPos.px} y2={xAxisPos.py} stroke="#cbd5e1" strokeWidth="1.2" />
            <line x1={yAxisNeg.px} y1={yAxisNeg.py} x2={yAxisPos.px} y2={yAxisPos.py} stroke="#cbd5e1" strokeWidth="1.2" />
            <line x1={zAxisNeg.px} y1={zAxisNeg.py} x2={zAxisPos.px} y2={zAxisPos.py} stroke="#cbd5e1" strokeWidth="1.2" />

            <text x={xAxisPos.px + 6} y={xAxisPos.py + 4} fontSize="10" fill="#64748b" fontFamily="monospace">
              Dim-1 (Domain)
            </text>
            <text x={yAxisPos.px - 28} y={yAxisPos.py - 8} fontSize="10" fill="#64748b" fontFamily="monospace">
              Dim-2 (Abstraction)
            </text>
            <text x={zAxisPos.px + 6} y={zAxisPos.py + 4} fontSize="10" fill="#64748b" fontFamily="monospace">
              Dim-3
            </text>

            {EMBEDDING_SIMILARITY_LINKS.map((link) => {
              const a = nodeMap[link.from];
              const b = nodeMap[link.to];
              if (!a || !b) return null;
              const midX = (a.proj.px + b.proj.px) / 2;
              const midY = (a.proj.py + b.proj.py) / 2;
              return (
                <g key={`${link.from}-${link.to}`}>
                  <line
                    x1={a.proj.px}
                    y1={a.proj.py}
                    x2={b.proj.px}
                    y2={b.proj.py}
                    stroke={link.color}
                    strokeWidth="2"
                    strokeDasharray="4 4"
                    opacity="0.75"
                  />
                  <rect x={midX - 24} y={midY - 9} width="48" height="16" rx="4" fill="#ffffff" stroke={link.color} strokeWidth="1" />
                  <text x={midX} y={midY + 2} textAnchor="middle" fontSize="9" fontWeight="bold" fill={link.color} fontFamily="monospace">
                    cos:{link.score}
                  </text>
                </g>
              );
            })}

            {projectedNodes.map((node) => {
              const isHovered = hoveredNode?.id === node.id;
              const radius = (isHovered ? 10 : 7.5) * node.proj.scale;
              return (
                <g
                  key={node.id}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className="cursor-pointer"
                >
                  <line
                    x1={origin.px}
                    y1={origin.py}
                    x2={node.proj.px}
                    y2={node.proj.py}
                    stroke={node.color}
                    strokeWidth={isHovered ? "1.5" : "0.75"}
                    opacity={isHovered ? "0.45" : "0.18"}
                  />
                  <circle cx={node.proj.px} cy={node.proj.py} r={radius + 5} fill={node.color} opacity={isHovered ? 0.25 : 0.12} />
                  <circle cx={node.proj.px} cy={node.proj.py} r={radius} fill={node.color} stroke="#ffffff" strokeWidth="2" />
                  <text
                    x={node.proj.px}
                    y={node.proj.py - radius - 6}
                    textAnchor="middle"
                    fontSize={isHovered ? "13" : "11.5"}
                    fontWeight="700"
                    fill="#0f172a"
                  >
                    {node.word}
                  </text>
                </g>
              );
            })}
          </svg>

          <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-white/90 backdrop-blur-xs border border-slate-200 text-[11px] text-slate-600 font-medium shadow-2xs">
            Drag canvas to rotate 3D vector space &bull; Hover nodes for coordinates
          </div>

          {hoveredNode && (
            <div className="absolute bottom-3 right-3 p-3 rounded-xl bg-white/95 backdrop-blur-xs border border-slate-200 shadow-md text-xs space-y-1 min-w-[205px]">
              <div className="flex items-center justify-between gap-2">
                <span className="font-bold text-slate-900 text-sm">&quot;{hoveredNode.word}&quot;</span>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-semibold text-white"
                  style={{ backgroundColor: hoveredNode.color }}
                >
                  {hoveredNode.cluster}
                </span>
              </div>
              <div className="text-slate-500 font-mono text-[11px]">
                3D PCA: <span className="text-slate-800 font-semibold">{hoveredNode.vec}</span>
              </div>
              <div className="text-slate-400 text-[10px]">Full Embedding: R^768 float32</div>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 pt-1">
          <div className="flex flex-wrap items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" /> Royalty Cluster
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-600" /> Deep Learning Cluster
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500" /> Architecture Cluster
            </span>
          </div>
          <span className="font-mono text-[11px] text-slate-500">cos(&theta;) = (A&middot;B) / (||A|| ||B||)</span>
        </div>
      </div>
    </div>
  );
}

const ATTN_TOKENS = ["The", "cat", "sat", "down"];
const RAW_ATTENTION_SCORES = [
  [0.72, 0.11, 0.09, 0.08],
  [0.34, 0.52, 0.09, 0.05],
  [0.10, 0.68, 0.17, 0.05],
  [0.06, 0.21, 0.63, 0.10]
];

function VisxAttentionHeatmapLab() {
  const [causalMask, setCausalMask] = useState(true);
  const [activeRow, setActiveRow] = useState(2);
  const [hoveredCell, setHoveredCell] = useState({ r: 2, c: 1 });

  const matrix = useMemo(() => {
    return RAW_ATTENTION_SCORES.map((row, rIdx) => {
      if (!causalMask) return row;
      const unmaskedSum = row.slice(0, rIdx + 1).reduce((acc, v) => acc + v, 0);
      return row.map((val, cIdx) => {
        if (cIdx > rIdx) return null;
        return Number((val / unmaskedSum).toFixed(2));
      });
    });
  }, [causalMask]);

  const activeWeights = matrix[activeRow];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md">
              Interactive Lab • @visx/heatmap
            </span>
            <h3 className="text-base font-extrabold text-[#17394D]">Causal Self-Attention Matrix</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Formula: <code className="font-mono text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded">Softmax(QKᵀ / &radic;dₖ + M)</code>
          </p>
        </div>

        <button
          type="button"
          onClick={() => setCausalMask((m) => !m)}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-bold border transition shadow-2xs ${
            causalMask
              ? "bg-indigo-600 text-white border-indigo-600"
              : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
          }`}
        >
          Apply Causal Mask (Decoder): {causalMask ? "ON" : "OFF"}
        </button>
      </div>

      <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        <div className="lg:col-span-7">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex justify-between px-2">
            <span>Rows: Query (Q)</span>
            <span>Columns: Key (K)</span>
          </div>

          <div className="grid grid-cols-5 gap-2 font-mono text-xs bg-slate-50 p-4 rounded-xl border border-slate-200/80">
            <div className="flex items-center justify-center text-[10px] text-slate-400 font-sans">Q \ K</div>
            {ATTN_TOKENS.map((tok, cIdx) => (
              <div
                key={`col-${tok}`}
                className={`text-center font-bold py-1.5 rounded-md transition ${
                  hoveredCell?.c === cIdx ? "bg-indigo-100 text-indigo-700" : "text-slate-600"
                }`}
              >
                {tok}
              </div>
            ))}

            {ATTN_TOKENS.map((rowTok, rIdx) => (
              <React.Fragment key={`row-${rowTok}`}>
                <button
                  type="button"
                  onClick={() => setActiveRow(rIdx)}
                  onMouseEnter={() => setActiveRow(rIdx)}
                  className={`flex items-center justify-end pr-2.5 font-bold rounded-md transition cursor-pointer ${
                    activeRow === rIdx
                      ? "bg-indigo-600 text-white shadow-2xs"
                      : "text-slate-700 hover:bg-slate-200/70"
                  }`}
                >
                  {rowTok}
                </button>

                {matrix[rIdx].map((val, cIdx) => {
                  const isMasked = val === null;
                  const isRowSelected = activeRow === rIdx;
                  const isCellHovered = hoveredCell?.r === rIdx && hoveredCell?.c === cIdx;

                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      onMouseEnter={() => {
                        setActiveRow(rIdx);
                        setHoveredCell({ r: rIdx, c: cIdx });
                      }}
                      style={{
                        backgroundColor: isMasked
                          ? "#f1f5f9"
                          : `rgba(79, 70, 229, ${Math.max(0.12, val * 0.95)})`
                      }}
                      className={`h-12 rounded-lg flex flex-col items-center justify-center border transition-all duration-200 cursor-pointer ${
                        isCellHovered
                          ? "ring-2 ring-indigo-600 scale-105 z-10 border-white"
                          : isRowSelected
                          ? "border-indigo-400"
                          : "border-slate-200/60"
                      } ${
                        isMasked
                          ? "text-slate-400"
                          : val > 0.45
                          ? "text-white font-bold"
                          : "text-slate-800 font-medium"
                      }`}
                    >
                      <span>{isMasked ? "-inf" : val.toFixed(2)}</span>
                      {isMasked && <span className="text-[9px] font-sans">masked</span>}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
            <div>
              <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Query Token Focus</span>
              <h4 className="text-sm font-bold text-[#17394D]">
                Token &quot;{ATTN_TOKENS[activeRow]}&quot; attends to:
              </h4>
            </div>
            <span className="px-2 py-1 rounded bg-indigo-100 text-indigo-700 font-mono text-xs font-bold">
              Pos #{activeRow}
            </span>
          </div>

          <div className="space-y-2.5">
            {ATTN_TOKENS.map((targetTok, idx) => {
              const weight = activeWeights[idx];
              const pct = weight === null ? 0 : Math.round(weight * 100);
              return (
                <div key={targetTok} className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className={weight === null ? "text-slate-400 line-through" : "font-semibold text-slate-700"}>
                      &rarr; &quot;{targetTok}&quot;
                    </span>
                    <span className={weight === null ? "text-slate-400" : "font-bold text-indigo-600"}>
                      {weight === null ? "0% (-inf masked)" : `${pct}%`}
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-600 rounded-full transition-all duration-300"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <p className="text-[11px] text-slate-500 leading-relaxed pt-1 border-t border-slate-200/80">
            {causalMask
              ? "With Causal Masking ON, tokens cannot attend to future positions (upper triangle = -inf), preventing data leakage during autoregressive training."
              : "With Causal Masking OFF (Encoder mode like BERT), every token attends bidirectionally to all tokens in the sequence."}
          </p>
        </div>
      </div>
    </div>
  );
}

const TRANSFORMER_FLOW_NODES = [
  {
    id: "embed",
    title: "Input Embeddings + Positional Encoding",
    subtitle: "Token Lookup + RoPE / Sinusoidal",
    shape: "[B, T, 768]",
    y: 24,
    accent: "border-slate-300 bg-white",
    badge: "bg-slate-100 text-slate-700",
    details: {
      operation: "X = W_e[tokens] + PE(pos)",
      params: "50,257 × 768 = 38.6M params",
      explanation: "Maps discrete integer token IDs into continuous 768-dimensional vectors and injects sequence order information."
    }
  },
  {
    id: "mha",
    title: "Multi-Head Self-Attention (12 Heads)",
    subtitle: "Causal Masked Scaled Dot-Product",
    shape: "[B, 12, T, 64] → [B, T, 768]",
    y: 118,
    accent: "border-indigo-400 bg-indigo-50/40",
    badge: "bg-indigo-100 text-indigo-700",
    details: {
      operation: "Attn(Q,K,V) = Softmax(QKᵀ/√64)V",
      params: "4 × (768 × 768) = 2.36M params",
      explanation: "Projects embeddings into 12 parallel attention heads (head_dim = 64) so each head tracks different syntactic/semantic relationships."
    }
  },
  {
    id: "norm1",
    title: "Add & LayerNorm (Residual Stream)",
    subtitle: "x + Dropout(Sublayer(x))",
    shape: "[B, T, 768]",
    y: 212,
    accent: "border-emerald-400 bg-emerald-50/40",
    badge: "bg-emerald-100 text-emerald-700",
    details: {
      operation: "LayerNorm(X + MultiHeadAttn(X))",
      params: "2 × 768 = 1,536 params (γ, β)",
      explanation: "Residual skip connections prevent vanishing gradients in deep networks while LayerNorm stabilizes activation variance."
    }
  },
  {
    id: "mlp",
    title: "Feed-Forward Network (MLP)",
    subtitle: "Expansion 4x with GELU / SwiGLU",
    shape: "768 → 3072 → 768",
    y: 306,
    accent: "border-amber-400 bg-amber-50/40",
    badge: "bg-amber-100 text-amber-800",
    details: {
      operation: "W₂ · GELU(W₁X + b₁) + b₂",
      params: "2 × (768 × 3072) = 4.72M params",
      explanation: "Acts as key-value factual memory storage, expanding token representations to 3072 dims before projecting back to 768."
    }
  }
];

function XYFlowArchitectureLab() {
  const [selectedNodeId, setSelectedNodeId] = useState("mha");
  const selectedNode = TRANSFORMER_FLOW_NODES.find((n) => n.id === selectedNodeId) || TRANSFORMER_FLOW_NODES[1];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-emerald-100 text-emerald-700 rounded-md">
              Interactive Lab • @xyflow/react
            </span>
            <h3 className="text-base font-extrabold text-[#17394D]">Transformer Block Architecture Flow</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Click any node in the flow graph (including the curved Residual +x highway) to inspect tensor shapes &amp; math.
          </p>
        </div>

        <span className="text-xs font-mono bg-slate-100 text-slate-700 px-3 py-1 rounded-lg border border-slate-200">
          GPT-2 Small Block (N = 12 Layers)
        </span>
      </div>

      <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div
          className="lg:col-span-7 relative min-h-[395px] rounded-xl border border-slate-200 overflow-hidden select-none"
          style={{
            backgroundColor: "#f8fafc",
            backgroundImage: "radial-gradient(#cbd5e1 1px, transparent 1px)",
            backgroundSize: "18px 18px"
          }}
        >
          <svg viewBox="0 0 460 395" className="w-full h-full">
            <defs>
              <marker id="arrowIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#4f46e5" />
              </marker>
              <marker id="arrowEmerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
              </marker>
            </defs>

            <line x1="200" y1="82" x2="200" y2="114" stroke="#4f46e5" strokeWidth="2.2" strokeDasharray="5 5" markerEnd="url(#arrowIndigo)" />
            <line x1="200" y1="176" x2="200" y2="208" stroke="#4f46e5" strokeWidth="2.2" strokeDasharray="5 5" markerEnd="url(#arrowIndigo)" />
            <line x1="200" y1="270" x2="200" y2="302" stroke="#4f46e5" strokeWidth="2.2" strokeDasharray="5 5" markerEnd="url(#arrowIndigo)" />

            <path
              d="M 355 54 C 430 54, 430 242, 358 242"
              fill="none"
              stroke="#059669"
              strokeWidth="2.2"
              strokeDasharray="6 4"
              markerEnd="url(#arrowEmerald)"
            />
            <rect x="366" y="136" width="84" height="22" rx="6" fill="#ecfdf5" stroke="#6ee7b7" />
            <text x="408" y="150" textAnchor="middle" fontSize="10" fontWeight="700" fill="#047857" fontFamily="monospace">
              Residual +x
            </text>
          </svg>

          <div className="absolute inset-0 pointer-events-none">
            {TRANSFORMER_FLOW_NODES.map((node) => {
              const isSelected = node.id === selectedNodeId;
              return (
                <button
                  key={node.id}
                  type="button"
                  onClick={() => setSelectedNodeId(node.id)}
                  style={{ top: `${node.y}px`, left: "24px", width: "326px" }}
                  className={`pointer-events-auto absolute text-left px-3.5 py-2.5 rounded-xl border-2 transition-all cursor-pointer shadow-2xs ${
                    node.accent
                  } ${
                    isSelected
                      ? "ring-2 ring-indigo-500/30 border-indigo-600 scale-[1.02] shadow-md"
                      : "hover:border-slate-400"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold text-[#17394D] truncate">{node.title}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold shrink-0 ${node.badge}`}>
                      {node.shape}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5">{node.subtitle}</div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-indigo-100 text-indigo-700">
                Layer Inspector
              </span>
              <span className="text-[11px] font-mono text-slate-400">d_model = 768</span>
            </div>

            <h4 className="text-base font-bold text-[#17394D]">{selectedNode.title}</h4>
            <p className="text-xs text-slate-600 leading-relaxed">{selectedNode.details.explanation}</p>

            <div className="space-y-2 pt-2">
              <div className="p-3 rounded-lg bg-white border border-slate-200">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Tensor Transformation</div>
                <div className="font-mono text-xs font-bold text-indigo-700 mt-0.5">{selectedNode.shape}</div>
              </div>

              <div className="p-3 rounded-lg bg-white border border-slate-200">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Mathematical Operation</div>
                <div className="font-mono text-xs font-semibold text-slate-800 mt-0.5">{selectedNode.details.operation}</div>
              </div>

              <div className="p-3 rounded-lg bg-white border border-slate-200">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Learnable Parameters</div>
                <div className="font-mono text-xs text-emerald-700 font-semibold mt-0.5">{selectedNode.details.params}</div>
              </div>
            </div>
          </div>

          <div className="text-[11px] text-slate-400 flex items-center justify-between pt-2 border-t border-slate-200">
            <span>Batch (B) = 1, Seq Len (T) = 4</span>
            <span>Pre-Norm GPT Block</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const CANDIDATE_LOGITS = [
  { token: "Paris", logit: 5.2 },
  { token: "Lyon", logit: 3.1 },
  { token: "France", logit: 2.3 },
  { token: "nice", logit: 1.4 },
  { token: "banana", logit: 0.1 }
];

function RechartsTrainingSamplingLab() {
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);

  const distribution = useMemo(() => {
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

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden flex flex-col">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-slate-50/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider bg-indigo-100 text-indigo-700 rounded-md">
              Interactive Lab • recharts
            </span>
            <h3 className="text-base font-extrabold text-[#17394D]">Temperature &amp; Nucleus (Top-P) Sampling</h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Formula: <code className="font-mono text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded">P(xᵢ) = exp(zᵢ / T) / &Sigma; exp(zⱼ / T)</code>
          </p>
        </div>

        <div className="px-3.5 py-1.5 rounded-xl bg-slate-900 text-slate-100 font-mono text-xs flex items-center gap-1.5 shadow-2xs">
          <span className="text-slate-400">Prompt:</span>
          <span>&quot;The capital of France is</span>
          <span className="px-1.5 py-0.5 rounded bg-indigo-500 text-white font-bold animate-pulse">___</span>
          <span>&quot;</span>
        </div>
      </div>

      <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        <div className="lg:col-span-5 bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-5">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-xs font-bold text-[#17394D]">Temperature (T)</label>
              <span className="px-2 py-0.5 rounded-md bg-indigo-100 text-indigo-700 font-mono text-xs font-bold">
                {temperature.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0.2"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-indigo-600 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>0.2 (Deterministic)</span>
              <span>1.0 (Balanced)</span>
              <span>2.0 (High Entropy)</span>
            </div>
          </div>

          <div className="space-y-2 pt-3 border-t border-slate-200">
            <div className="flex justify-between items-center">
              <label className="text-xs font-bold text-[#17394D]">Top-P (Nucleus Cutoff)</label>
              <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-700 font-mono text-xs font-bold">
                {topP.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="1.00"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full accent-emerald-600 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>0.10 (Strict)</span>
              <span>0.90 (Standard)</span>
              <span>1.00 (All Tokens)</span>
            </div>
          </div>

          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={() => {
                setTemperature(0.2);
                setTopP(0.5);
              }}
              className="flex-1 py-1.5 text-[11px] font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-100 transition"
            >
              Factual Coding
            </button>
            <button
              type="button"
              onClick={() => {
                setTemperature(1.5);
                setTopP(0.95);
              }}
              className="flex-1 py-1.5 text-[11px] font-semibold rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-100 transition"
            >
              Creative Writing
            </button>
          </div>
        </div>

        <div className="lg:col-span-7 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-500 px-1">
            <span className="font-semibold text-[#17394D]">Next-Token Softmax Probability Distribution</span>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-xs bg-indigo-600" /> Active Nucleus
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-xs bg-slate-300" /> Filtered Out
              </span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3">
            {distribution.map((row) => (
              <div key={row.token} className="space-y-1">
                <div className="flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-bold px-2 py-0.5 rounded ${
                        row.inNucleus ? "bg-white text-slate-900 border border-slate-200" : "text-slate-400 line-through"
                      }`}
                    >
                      &quot;{row.token}&quot;
                    </span>
                    <span className="text-[10px] text-slate-400">logit: {row.logit.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {!row.inNucleus && (
                      <span className="text-[10px] font-sans text-slate-400 bg-slate-200/70 px-1.5 py-0.5 rounded">
                        Cut by Top-P
                      </span>
                    )}
                    <span className={`font-bold w-12 text-right ${row.inNucleus ? "text-indigo-700" : "text-slate-400"}`}>
                      {row.pct}%
                    </span>
                  </div>
                </div>

                <div className="w-full h-5 bg-slate-200/70 rounded-lg overflow-hidden p-0.5">
                  <div
                    style={{ width: `${Math.max(2, row.pct)}%` }}
                    className={`h-full rounded-md transition-all duration-300 ${
                      row.inNucleus
                        ? row.token === "Paris"
                          ? "bg-indigo-600"
                          : "bg-emerald-500"
                        : "bg-slate-300"
                    }`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ConceptAnimationLab({ animType }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    setStep(0);
    setPlaying(true);
  }, [animType]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setStep((s) => (s + 1) % 4);
    }, 1700);
    return () => clearInterval(timer);
  }, [playing]);

  if (animType === "bpe_merge") return <MotionBPELab />;
  if (animType === "embeddings_pos") return <ThreeDEmbeddingLab />;
  if (animType === "qkv_attention" || animType === "causal_mask") return <VisxAttentionHeatmapLab />;
  if (animType === "transformer_block") return <XYFlowArchitectureLab />;
  if (animType === "lr_cosine" || animType === "softmax_logits") return <RechartsTrainingSamplingLab />;

  const controls = (label) => (
    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 mb-4 border-b border-slate-200/80">
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-ping" />
        <span className="text-xs font-extrabold uppercase tracking-wider text-[#17394D]">{label}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-[#17394D]"
        >
          {playing ? "Pause" : "Auto-Play"}
        </button>
        {[0, 1, 2, 3].map((idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => {
              setPlaying(false);
              setStep(idx);
            }}
            className={`w-6 h-6 rounded-md text-[11px] font-mono font-bold transition-all ${
              step === idx ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {idx + 1}
          </button>
        ))}
      </div>
    </div>
  );

  if (animType === "python_oop") {
    const steps = [
      { title: "Step 1: Data Structures (list/tuple/dict/set)", badge: "vocab: dict[str, int] | shape: tuple", desc: "Lists hold token IDs, tuples hold fixed (B,T,C) shapes, dicts map tokens, and sets track unique items." },
      { title: "Step 2: @dataclass Config Blueprint", badge: "cfg = GPTConfig(emb_dim=384, n_heads=6)", desc: "Bundles all architecture knobs into one clean, readable configuration object." },
      { title: "Step 3: Token Dict Lookup (O(1))", badge: 'vocab["Paris"] -> 6342', desc: "Python dictionaries map words to integer IDs instantaneously." },
      { title: "Step 4: Stateful nn.Module Class", badge: "model = GPTModel(cfg)", desc: "Instantiates every layer and registers learnable weight matrices automatically." }
    ];
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Mental Model: Python OOP & .venv Architecture")}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-2.5">
          {steps.map((s, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-xl border transition-all duration-300 ${
                step === idx
                  ? "bg-gradient-to-br from-teal-50 to-indigo-50 border-indigo-500 scale-[1.02] shadow-xs"
                  : "bg-slate-50/70 border-slate-200/80 opacity-65"
              }`}
            >
              <div className="text-[11px] font-bold text-[#17394D]">{s.title}</div>
              <code className="block text-[10px] font-mono bg-white border border-slate-200 rounded px-1.5 py-1 my-1.5 text-indigo-700 truncate">
                {s.badge}
              </code>
              <p className="text-[11px] text-slate-600 leading-snug">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (animType === "broadcasting") {
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Visual: NumPy Broadcasting (4×3 Matrix + 1×3 Bias)")}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 py-2">
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-center">
            <div className="text-[11px] font-mono font-bold text-slate-500 mb-1.5">Token Matrix X (4 × 3)</div>
            <div className="grid grid-cols-3 gap-1 font-mono text-xs">
              {[1.0, 0.2, -0.5, 0.9, 0.1, -0.4, -0.8, 0.7, 0.3, 0.0, -0.5, 0.9].map((v, i) => (
                <span key={i} className="px-2 py-1 rounded bg-white border border-slate-200 text-[#17394D]">
                  {v.toFixed(1)}
                </span>
              ))}
            </div>
          </div>
          <span className="text-xl font-extrabold text-indigo-600">+</span>
          <div className="p-3 rounded-xl bg-teal-50/70 border border-teal-300 text-center">
            <div className="text-[11px] font-mono font-bold text-teal-800 mb-1.5">
              Bias (3,) &rarr; Stretched to (4 × 3)
            </div>
            <div className="space-y-1 font-mono text-xs">
              {[0, 1, 2, 3].map((row) => (
                <div
                  key={row}
                  className={`grid grid-cols-3 gap-1 transition-all duration-300 ${
                    row === 0 || step >= 1 ? "opacity-100" : "opacity-25 scale-95"
                  }`}
                >
                  {[0.1, -0.1, 0.5].map((b, c) => (
                    <span
                      key={c}
                      className={`px-2 py-1 rounded border ${
                        row === 0
                          ? "bg-teal-600 text-white border-teal-700 font-bold"
                          : "bg-teal-100/80 text-teal-900 border-teal-300 border-dashed"
                      }`}
                    >
                      {b.toFixed(1)}
                    </span>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <p className="text-xs text-center text-slate-600 mt-2">
          {step === 0
            ? "1D bias vector starts as a single row of 3 numbers."
            : "NumPy automatically broadcasts that single row across all 4 token rows without copying memory!"}
        </p>
      </div>
    );
  }

  if (animType === "tensor_btc") {
    const stages = [
      { label: "Input Token IDs", shape: "(B=2, T=5)", note: "2 sentences, 5 word IDs each" },
      { label: "After Embedding", shape: "(B=2, T=5, C=64)", note: "Each token becomes a 64-float vector" },
      { label: "Split 4 Heads", shape: "(B=2, H=4, T=5, d=16)", note: "64 dims split into 4 parallel spotlights" },
      { label: "LM Head Logits", shape: "(B=2, T=5, V=50257)", note: "Scores every word in vocabulary!" }
    ];
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Tensor Shape Tracer: (Batch, Tokens, Channels)")}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {stages.map((st, idx) => (
            <div
              key={idx}
              className={`p-3.5 rounded-xl border transition-all duration-300 ${
                step === idx
                  ? "bg-indigo-600 text-white border-indigo-700 shadow-sm scale-[1.03]"
                  : "bg-slate-50 text-[#17394D] border-slate-200"
              }`}
            >
              <div className="text-[11px] font-bold opacity-85">{st.label}</div>
              <div className="font-mono text-xs font-extrabold my-1.5">{st.shape}</div>
              <div className="text-[11px] opacity-85 leading-snug">{st.note}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (animType === "autograd_graph") {
    const nodes = [
      { id: "w=3, x=2", val: "z = 7.0", grad: "dL/dw = -12.0" },
      { id: "target y=10", val: "diff = -3.0", grad: "dL/ddiff = -6.0" },
      { id: "Loss = diff²", val: "Loss = 9.0", grad: "dL/dL = 1.0" }
    ];
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Backpropagation Flow: Forward Pass vs. Backward Gradients")}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {nodes.map((n, idx) => {
            const isForward = step <= 1 && step === idx;
            const isBackward = step >= 2 && 3 - step === idx;
            return (
              <div
                key={n.id}
                className={`p-3.5 rounded-xl border transition-all duration-300 ${
                  isBackward
                    ? "bg-rose-50 border-rose-500 shadow-xs"
                    : isForward
                      ? "bg-teal-50 border-teal-500 shadow-xs"
                      : "bg-slate-50 border-slate-200"
                }`}
              >
                <div className="text-[11px] font-mono text-slate-500">{n.id}</div>
                <div className="text-sm font-extrabold text-[#17394D] my-1">{n.val}</div>
                <div className="text-xs font-mono font-bold text-rose-700">{n.grad}</div>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-slate-600 mt-3 text-center">
          {step <= 1
            ? "Forward Pass (left -> right): computing prediction z=7.0 and squared error Loss=9.0."
            : "Backward Pass (right -> left): Chain Rule multiplies gradients backward so weight w knows to increase!"}
        </p>
      </div>
    );
  }

  if (animType === "gradient_descent") {
    const points = [
      { stepLabel: "Step 0 (Random)", w: -3.5, loss: 12.25 },
      { stepLabel: "Step 25", w: -1.8, loss: 3.24 },
      { stepLabel: "Step 60", w: -0.6, loss: 0.36 },
      { stepLabel: "Step 120 (Converged)", w: 0.0, loss: 0.01 }
    ];
    const cur = points[step];
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Optimization: Stepping Down the Loss Valley")}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {points.map((pt, idx) => (
            <div
              key={pt.stepLabel}
              className={`p-3 rounded-xl border transition-all ${
                step === idx ? "bg-amber-50 border-amber-500 shadow-xs" : "bg-slate-50 border-slate-200 opacity-70"
              }`}
            >
              <div className="text-[11px] font-bold text-[#17394D]">{pt.stepLabel}</div>
              <div className="font-mono text-xs text-indigo-700 mt-1">Weight offset: {pt.w}</div>
              <div className="font-mono text-xs font-extrabold text-amber-800">Loss: {pt.loss}</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-center text-slate-600 mt-2.5">
          Current State: <strong>{cur.stepLabel}</strong> &mdash; AdamW nudges weights downhill until Loss approaches 0.
        </p>
      </div>
    );
  }

  if (animType === "sliding_window") {
    const stream = ["The", "capital", "of", "France", "is", "Paris", "<|eos|>"];
    const start = Math.min(step, 2);
    const xSlice = stream.slice(start, start + 4);
    const ySlice = stream.slice(start + 1, start + 5);
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated Sliding-Window Next-Token Dataset (X vs. Shifted Y)")}
        <div className="space-y-2.5 font-mono text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="w-24 font-bold text-indigo-700">Input X:</span>
            {xSlice.map((t, i) => (
              <span key={i} className="px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-300 text-indigo-900 font-bold">
                {t}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="w-24 font-bold text-teal-700">Target Y (+1):</span>
            {ySlice.map((t, i) => (
              <span key={i} className="px-2.5 py-1 rounded-lg bg-teal-50 border border-teal-300 text-teal-900 font-bold">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (animType === "lora_lowrank") {
    return (
      <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
        {controls("Animated LoRA Adapter: Frozen Brain + Tiny Trainable A @ B Matrices")}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3.5 rounded-xl bg-sky-50 border border-sky-200">
            <div className="font-extrabold text-sky-950">Base Pretrained Matrix W (4096 × 4096)</div>
            <div className="font-mono text-sky-800 mt-1">16,777,216 params &bull; FROZEN (requires_grad=False)</div>
          </div>
          <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-300">
            <div className="font-extrabold text-emerald-950">LoRA A (4096×8) @ B (8×4096)</div>
            <div className="font-mono text-emerald-800 mt-1">65,536 params (0.39%) &bull; TRAINABLE!</div>
          </div>
        </div>
      </div>
    );
  }

  const hops = [
    "1. Client sends JSON: {'prompt': 'Explain DNS'}",
    "2. Pydantic validates types & FastAPI routes request",
    "3. PyTorch GPT runs inside torch.no_grad() on GPU/CPU",
    "4. Uvicorn returns JSON response in milliseconds!"
  ];
  return (
    <div className="bg-white rounded-2xl border border-indigo-200/90 p-5 shadow-xs">
      {controls("Animated Local LLM Inference Request Lifecycle")}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {hops.map((h, idx) => (
          <div
            key={h}
            className={`p-3 rounded-xl border text-xs font-bold transition-all ${
              step === idx ? "bg-teal-600 text-white border-teal-700" : "bg-slate-50 text-[#17394D] border-slate-200"
            }`}
          >
            {h}
          </div>
        ))}
      </div>
    </div>
  );
}

// =========================================================================
// 3. BUILT-IN VITE.JS ZIP BUNDLE GENERATOR (NO EXTERNAL CDN REQUIRED)
// =========================================================================
function createSingleZipBlob(files) {
  const encoder = new TextEncoder();
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[i] = c;
  }
  const crc32 = (bytes) => {
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  };

  const localParts = [];
  const centralParts = [];
  let offset = 0;

  files.forEach(({ path, content }) => {
    const nameBytes = encoder.encode(path);
    const dataBytes = encoder.encode(content);
    const crc = crc32(dataBytes);

    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(localHeader.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(6, 0, true);
    lv.setUint16(8, 0, true);
    lv.setUint16(10, 0, true);
    lv.setUint16(12, 0, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, dataBytes.length, true);
    lv.setUint32(22, dataBytes.length, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);

    localParts.push(localHeader, dataBytes);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, dataBytes.length, true);
    cv.setUint32(24, dataBytes.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);

    centralParts.push(centralHeader);
    offset += localHeader.length + dataBytes.length;
  });

  const centralSize = centralParts.reduce((acc, p) => acc + p.length, 0);
  const endRecord = new Uint8Array(22);
  const ev = new DataView(endRecord.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  ev.setUint16(20, 0, true);

  return new Blob([...localParts, ...centralParts, endRecord], { type: "application/zip" });
}

function sanitizeExportedFnSource(fnSource) {
  // Replace esbuild/bundler-mangled React references (e.g., import_react.useState or import_react.default.createElement)
  return fnSource
    .replace(/\bimport_react\d*\.default\b/g, "React")
    .replace(/\bimport_react\d*\./g, "React.");
}

function downloadViteProjectZip() {
  const pkgJson = JSON.stringify(
    {
      name: "llm-from-scratch-academy",
      private: true,
      version: "1.0.0",
      type: "module",
      scripts: { dev: "vite", build: "vite build", preview: "vite preview" },
      dependencies: {
        react: "^18.3.1",
        "react-dom": "^18.3.1",
        motion: "^11.11.17",
        three: "^0.169.0",
        "@react-three/fiber": "^8.17.10",
        "@visx/heatmap": "^3.12.0",
        "@xyflow/react": "^12.3.5",
        recharts: "^2.13.3"
      },
      devDependencies: {
        "@vitejs/plugin-react": "^4.3.1",
        autoprefixer: "^10.4.20",
        postcss: "^8.4.47",
        tailwindcss: "^3.4.14",
        vite: "^5.4.8"
      }
    },
    null,
    2
  );

  const viteConfig = `import { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({\n  plugins: [react()],\n});\n`;
  const tailwindConfig = `export default {\n  content: ['./index.html', './src/**/*.{js,jsx}'],\n  theme: { extend: {} },\n  plugins: [],\n};\n`;
  const postcssConfig = `export default {\n  plugins: { tailwindcss: {}, autoprefixer: {} },\n};\n`;
  const indexCss = `@tailwind base;\n@tailwind components;\n@tailwind utilities;\n`;
  const mainJsx = `import React from 'react';\nimport ReactDOM from 'react-dom/client';\nimport App from './App.jsx';\nimport './index.css';\n\nReactDOM.createRoot(document.getElementById('root')).render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>\n);\n`;
  const indexHtml = `<!doctype html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n    <title>LLM From Scratch — Interactive Learning Academy</title>\n  </head>\n  <body>\n    <div id="root"></div>\n    <script type="module" src="/src/main.jsx"></script>\n  </body>\n</html>\n`;
  const netlifyToml = `[build]\n  command = "npm run build"\n  publish = "dist"\n\n[[redirects]]\n  from = "/*"\n  to = "/index.html"\n  status = 200\n`;
  const vercelJson = JSON.stringify({ rewrites: [{ source: "/(.*)", destination: "/index.html" }] }, null, 2);
  const readme = `# LLM From Scratch — Self-Contained Multi-Page Learning Academy\n\n## Run Locally\n\`\`\`bash\nnpm install\nnpm run dev\n\`\`\`\n\n## Deploy to Netlify or Vercel\nPush this folder to GitHub and import into Netlify or Vercel (build command: \`npm run build\`, publish directory: \`dist\`).\n`;

  const appSource = [
    'import React, { useState, useEffect, useMemo, useRef } from "react";',
    "const PIPELINE_STAGES = " + JSON.stringify(PIPELINE_STAGES, null, 2) + ";",
    "const SETUP_SOFTWARES = " + JSON.stringify(SETUP_SOFTWARES, null, 2) + ";",
    "const SETUP_PACKAGES = " + JSON.stringify(SETUP_PACKAGES, null, 2) + ";",
    "const SETUP_COMMANDS_TEXT = " + JSON.stringify(SETUP_COMMANDS_TEXT) + ";",
    "const ROADMAP_PHASES = " + JSON.stringify(ROADMAP_PHASES, null, 2) + ";",
    "const ALL_TOPICS = ROADMAP_PHASES.flatMap((phase) => phase.topics.map((t) => ({ ...t, phaseId: phase.id, phaseNumber: phase.number, phaseTitle: phase.title })));",
    'const STORAGE_KEY = "llm_scratch_roadmap_completed_v2";',
    sanitizeExportedFnSource(getAccentClasses.toString()),
    sanitizeExportedFnSource(MotionBPELab.toString()),
    sanitizeExportedFnSource(ThreeDEmbeddingLab.toString()),
    sanitizeExportedFnSource(VisxAttentionHeatmapLab.toString()),
    sanitizeExportedFnSource(XYFlowArchitectureLab.toString()),
    sanitizeExportedFnSource(RechartsTrainingSamplingLab.toString()),
    sanitizeExportedFnSource(ConceptAnimationLab.toString()),
    sanitizeExportedFnSource(App.toString().replace("export default function App()", "function App()")),
    "export default App;"
  ].join("\n\n");

  const zipBlob = createSingleZipBlob([
    { path: "llm-from-scratch-academy/package.json", content: pkgJson },
    { path: "llm-from-scratch-academy/vite.config.js", content: viteConfig },
    { path: "llm-from-scratch-academy/tailwind.config.js", content: tailwindConfig },
    { path: "llm-from-scratch-academy/postcss.config.js", content: postcssConfig },
    { path: "llm-from-scratch-academy/index.html", content: indexHtml },
    { path: "llm-from-scratch-academy/netlify.toml", content: netlifyToml },
    { path: "llm-from-scratch-academy/vercel.json", content: vercelJson },
    { path: "llm-from-scratch-academy/README.md", content: readme },
    { path: "llm-from-scratch-academy/src/index.css", content: indexCss },
    { path: "llm-from-scratch-academy/src/main.jsx", content: mainJsx },
    { path: "llm-from-scratch-academy/src/App.jsx", content: appSource }
  ]);

  const url = URL.createObjectURL(zipBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "llm-from-scratch-academy-vite.zip";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// =========================================================================
// 4. MAIN REACT APPLICATION (HOME ROADMAP PAGE + DEDICATED MODULE PAGES)
// =========================================================================
export default function App() {
  const [completedIds, setCompletedIds] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const parsed = saved ? JSON.parse(saved) : [];
      return new Set(Array.isArray(parsed) ? parsed : []);
    } catch {
      return new Set();
    }
  });

  const [currentFilter, setCurrentFilter] = useState("all");
  const [collapsedPhases, setCollapsedPhases] = useState(() => new Set());
  const [activeStageId, setActiveStageId] = useState(1);
  const [activeTopicId, setActiveTopicId] = useState(null);
  const [quizChoice, setQuizChoice] = useState(null);
  const [copiedCode, setCopiedCode] = useState(false);
  const [scrollSpinePct, setScrollSpinePct] = useState(0);
  const [toast, setToast] = useState(null);
  const [copiedSetup, setCopiedSetup] = useState(false);
  const timelineRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...completedIds]));
    } catch {
      // Ignore storage quota errors
    }
  }, [completedIds]);

  useEffect(() => {
    setQuizChoice(null);
    setCopiedCode(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [activeTopicId]);

  useEffect(() => {
    const handleScroll = () => {
      const el = timelineRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const windowH = window.innerHeight;
      const totalScrollable = rect.height - windowH * 0.4;
      if (totalScrollable <= 0) {
        setScrollSpinePct(100);
        return;
      }
      const scrolled = Math.min(Math.max(windowH * 0.5 - rect.top, 0), totalScrollable);
      setScrollSpinePct(Math.round((scrolled / totalScrollable) * 100));
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
  const masteryPct = Math.round((doneCount / totalTopics) * 100);

  const activeStage = PIPELINE_STAGES.find((s) => s.id === activeStageId) || PIPELINE_STAGES[0];
  const activeTopicIndex = useMemo(
    () => ALL_TOPICS.findIndex((t) => t.id === activeTopicId),
    [activeTopicId]
  );
  const activeTopic = activeTopicIndex >= 0 ? ALL_TOPICS[activeTopicIndex] : null;

  const filteredPhases = useMemo(() => {
    if (currentFilter === "all") return ROADMAP_PHASES;
    return ROADMAP_PHASES.filter((p) => p.id === currentFilter);
  }, [currentFilter]);

  const toggleTopic = (topicId) => {
    setCompletedIds((prev) => {
      const next = new Set(prev);
      if (next.has(topicId)) next.delete(topicId);
      else next.add(topicId);
      return next;
    });
  };

  const togglePhaseCollapse = (phaseId) => {
    setCollapsedPhases((prev) => {
      const next = new Set(prev);
      if (next.has(phaseId)) next.delete(phaseId);
      else next.add(phaseId);
      return next;
    });
  };

  const handleJumpToPhase = (phaseId) => {
    setActiveTopicId(null);
    if (currentFilter !== "all") setCurrentFilter("all");
    setCollapsedPhases((prev) => {
      const next = new Set(prev);
      next.delete(phaseId);
      return next;
    });
    setTimeout(() => {
      const target = document.getElementById(phaseId);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  };

  return (
    <div className="min-h-screen bg-[#F4F7FA] text-[#17394D] font-sans antialiased selection:bg-teal-100 selection:text-teal-900">
      {/* Sticky Top Header */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200/80 transition-all">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => setActiveTopicId(null)}
            className="flex items-center gap-2.5 text-left group"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-600 to-indigo-600 flex items-center justify-center text-white shadow-sm">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="6" cy="6" r="2.5" />
                <circle cx="18" cy="6" r="2.5" />
                <circle cx="12" cy="18" r="2.5" />
                <path strokeLinecap="round" d="M8.2 7.2L10.8 15.8M15.8 7.2L13.2 15.8M8.5 6h7" />
              </svg>
            </div>
            <div>
              <span className="font-bold text-[#17394D] group-hover:text-indigo-700 tracking-tight text-sm sm:text-base block leading-none">
                LLM From Scratch Academy
              </span>
              <span className="text-[11px] font-medium text-slate-500">
                {activeTopic ? `Module ${activeTopic.codeNum} Lesson Page` : "Self-Contained Interactive Course"}
              </span>
            </div>
          </button>

          <div className="flex items-center gap-2.5 sm:gap-3 flex-1 max-w-xl justify-end">
            <div className="flex-1 hidden md:block max-w-[200px]">
              <div className="flex justify-between text-xs font-semibold mb-1">
                <span className="text-slate-600">Mastery</span>
                <span className="text-teal-700 font-mono">
                  {masteryPct}% ({doneCount}/{totalTopics})
                </span>
              </div>
              <div className="w-full h-2 bg-slate-200/80 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${masteryPct}%` }}
                />
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                downloadViteProjectZip();
                setToast({ message: "Downloaded llm-from-scratch-academy-vite.zip!", confirm: false });
                setTimeout(() => setToast(null), 3200);
              }}
              className="inline-flex items-center gap-1.5 text-xs font-bold text-white bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 px-3 py-2 rounded-xl shadow-xs transition-all shrink-0"
            >
              <span>Download Vite Project (.zip)</span>
            </button>

            <button
              type="button"
              onClick={() => setToast({ message: "Reset all saved roadmap checkmarks?", confirm: true })}
              title="Reset saved progress"
              className="text-xs font-medium text-slate-500 hover:text-rose-600 px-2 py-1.5 rounded-lg hover:bg-rose-50 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      {/* ===================================================================== */}
      {/* VIEW A: DEDICATED FULL-PAGE MODULE LESSON VIEW                        */}
      {/* ===================================================================== */}
      {activeTopic ? (
        <main className="max-w-5xl mx-auto px-4 sm:px-6 py-8 pb-24 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => setActiveTopicId(null)}
              className="inline-flex items-center gap-2 text-xs sm:text-sm font-bold text-[#17394D] hover:text-indigo-700 bg-white border border-slate-200 px-4 py-2 rounded-xl shadow-2xs transition-all"
            >
              <span>&larr; Back to All 6 Phases</span>
            </button>

            <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
              <span>Phase {activeTopic.phaseNumber}</span>
              <span>/</span>
              <span className="font-bold text-indigo-700">Module {activeTopic.codeNum}</span>
              <span>&bull;</span>
              <span>{activeTopic.duration}</span>
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="text-xs font-bold uppercase tracking-wider px-3 py-1 rounded-full bg-teal-50 text-teal-800 border border-teal-200">
                Phase {activeTopic.phaseNumber}: {activeTopic.phaseTitle}
              </span>
              <label className="flex items-center gap-2 cursor-pointer select-none px-3.5 py-1.5 rounded-xl bg-slate-50 border border-slate-200 hover:border-teal-400">
                <input
                  type="checkbox"
                  checked={completedIds.has(activeTopic.id)}
                  onChange={() => toggleTopic(activeTopic.id)}
                  className="w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
                />
                <span className="text-xs font-bold text-[#17394D]">
                  {completedIds.has(activeTopic.id) ? "Completed ✓" : "Mark Module Completed"}
                </span>
              </label>
            </div>

            <h1 className="text-2xl sm:text-4xl font-extrabold text-[#17394D] tracking-tight">
              {activeTopic.codeNum} — {activeTopic.title}
            </h1>

            <p className="text-sm sm:text-base text-slate-700 leading-relaxed bg-sky-50/75 border border-sky-200/80 rounded-xl p-4">
              <strong className="text-sky-950">Plain-English Definition: </strong>
              {activeTopic.plainEnglish}
            </p>
          </div>

          {/* Step-by-Step VS Code Hands-On Lab (Where to Start, File to Create, Run & Output) */}
          {activeTopic.vsCodeLab && (
            <div className="bg-white rounded-2xl border-2 border-teal-500/40 shadow-sm overflow-hidden">
              <div className="px-6 py-4 bg-gradient-to-r from-teal-50 via-emerald-50/60 to-indigo-50/40 border-b border-teal-200/80 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-0.5 text-[11px] font-extrabold uppercase tracking-wider bg-teal-600 text-white rounded-md">
                      Start Here in VS Code &bull; Hands-On Lab
                    </span>
                    <span className="font-mono text-xs font-bold text-[#17394D] bg-white px-2.5 py-0.5 rounded-md border border-teal-200">
                      {activeTopic.vsCodeLab.folder}/{activeTopic.vsCodeLab.fileName}
                    </span>
                  </div>
                  <h2 className="text-base sm:text-lg font-extrabold text-[#17394D] mt-1">
                    Exact File Setup, Starter Code &amp; Terminal Execution Guide
                  </h2>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard?.writeText(activeTopic.vsCodeLab.starterCode);
                    setCopiedCode(true);
                    setTimeout(() => setCopiedCode(false), 2000);
                  }}
                  className="text-xs font-bold px-3.5 py-2 rounded-xl bg-[#17394D] hover:bg-indigo-900 text-white shadow-2xs transition-colors"
                >
                  {copiedCode ? "Copied Starter File!" : `Copy ${activeTopic.vsCodeLab.fileName}`}
                </button>
              </div>

              <div className="p-6 space-y-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {activeTopic.vsCodeLab.steps.map((st, i) => (
                    <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-50 border border-slate-200/85 text-xs text-[#17394D]">
                      <span className="w-5 h-5 rounded-full bg-teal-600 text-white font-mono font-bold text-[11px] flex items-center justify-center shrink-0 mt-0.5">
                        {i + 1}
                      </span>
                      <span className="leading-relaxed">{st}</span>
                    </div>
                  ))}
                </div>

                <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 text-slate-100 space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-teal-400 font-bold">
                      File: {activeTopic.vsCodeLab.folder}/{activeTopic.vsCodeLab.fileName}
                    </span>
                    <span className="text-slate-400">Python 3.11 (.venv)</span>
                  </div>
                  <pre className="text-xs font-mono overflow-x-auto leading-relaxed">
                    <code>{activeTopic.vsCodeLab.starterCode}</code>
                  </pre>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  <div className="rounded-xl bg-slate-950 border border-slate-800 p-3.5 text-xs font-mono">
                    <div className="text-[10px] uppercase tracking-wider text-amber-400 font-bold mb-1">
                      1. Run in VS Code Terminal (.venv active)
                    </div>
                    <div className="text-emerald-300 select-all">$ {activeTopic.vsCodeLab.runCmd}</div>
                  </div>
                  <div className="rounded-xl bg-emerald-50/70 border border-emerald-200 p-3.5 text-xs font-mono">
                    <div className="text-[10px] uppercase tracking-wider text-emerald-900 font-bold mb-1">
                      2. Expected Terminal Output
                    </div>
                    <pre className="text-emerald-950 whitespace-pre-wrap leading-snug">{activeTopic.vsCodeLab.expectedOutput}</pre>
                  </div>
                </div>
              </div>
            </div>
          )}

          <ConceptAnimationLab animType={activeTopic.animType} />

          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6">
            <h2 className="text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-3">
              Key Terms &amp; Jargon You Will Master Here
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {activeTopic.jargonTerms.map((j) => (
                <div key={j.term} className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 text-xs">
                  <span className="font-mono font-bold text-indigo-700 block mb-0.5">{j.term}</span>
                  <span className="text-slate-700 leading-snug">{j.def}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-5">
            <div className="border-b border-slate-100 pb-3">
              <span className="text-[11px] font-extrabold uppercase tracking-wider text-teal-700">
                Complete In-App Lesson Guide
              </span>
              <h2 className="text-xl font-extrabold text-[#17394D] mt-0.5">
                Core Concepts &amp; Step-by-Step Intuition
              </h2>
            </div>

            <div className="bg-teal-50/60 border border-teal-200/80 rounded-xl p-4 text-sm text-[#17394D]">
              <strong className="text-teal-950">Why It Matters for Building LLMs: </strong>
              {activeTopic.why}
            </div>

            {(activeTopic.lessonSections || []).map((sec, idx) => (
              <div key={idx} className="space-y-2.5 pt-2 border-t border-slate-100 first:border-t-0 first:pt-0">
                <h3 className="text-base font-bold text-[#17394D]">{sec.heading}</h3>
                <p className="text-sm text-slate-700 leading-relaxed">{sec.body}</p>
                {sec.codeExample && (
                  <pre className="bg-slate-900 text-slate-100 rounded-xl p-4 text-xs font-mono overflow-x-auto leading-relaxed border border-slate-800">
                    <code>{sec.codeExample}</code>
                  </pre>
                )}
              </div>
            ))}

            <div className="pt-2 border-t border-slate-100">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500 mb-2.5">
                Learning Checklist for Module {activeTopic.codeNum}
              </h3>
              <ul className="space-y-2">
                {activeTopic.concepts.map((c, idx) => (
                  <li key={idx} className="flex items-start gap-2.5 text-sm text-[#17394D]">
                    <span className="w-5 h-5 rounded-full bg-teal-50 text-teal-700 border border-teal-200 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                      &#10003;
                    </span>
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {activeTopic.liveExample && (
            <div className="bg-gradient-to-br from-violet-50/90 to-indigo-50/60 border border-violet-200/90 rounded-2xl p-6 shadow-2xs">
              <h2 className="text-xs font-extrabold uppercase tracking-wider text-violet-950 mb-1.5 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-violet-600" />
                Where Live Running AIs Use This Exact Concept
              </h2>
              <p className="text-sm text-[#17394D] leading-relaxed">{activeTopic.liveExample}</p>
            </div>
          )}

          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 text-slate-100 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-teal-400">
                Hands-On Python 3.11 / PyTorch Reference Code
              </span>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard?.writeText(activeTopic.code);
                  setCopiedCode(true);
                  setTimeout(() => setCopiedCode(false), 2000);
                }}
                className="text-xs font-semibold px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
              >
                {copiedCode ? "Copied!" : "Copy Code"}
              </button>
            </div>
            <pre className="text-xs sm:text-sm font-mono overflow-x-auto leading-relaxed">
              <code>{activeTopic.code}</code>
            </pre>
          </div>

          {activeTopic.quiz && (
            <div className="bg-white rounded-2xl border border-indigo-200/90 shadow-sm p-6 space-y-3">
              <span className="text-[11px] font-extrabold uppercase tracking-wider text-indigo-700">
                Quick Knowledge Check
              </span>
              <h3 className="text-base font-bold text-[#17394D]">{activeTopic.quiz.question}</h3>
              <div className="space-y-2">
                {activeTopic.quiz.options.map((opt, idx) => {
                  const chosen = quizChoice === idx;
                  const isCorrect = idx === activeTopic.quiz.answerIndex;
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setQuizChoice(idx)}
                      className={`w-full text-left p-3.5 rounded-xl border text-xs sm:text-sm font-medium transition-all ${
                        chosen
                          ? isCorrect
                            ? "bg-emerald-50 border-emerald-500 text-emerald-950 font-bold"
                            : "bg-rose-50 border-rose-500 text-rose-950"
                          : "bg-slate-50 hover:bg-white border-slate-200 text-[#17394D]"
                      }`}
                    >
                      {opt}
                    </button>
                  );
                })}
              </div>
              {quizChoice !== null && (
                <div
                  className={`p-3 rounded-xl text-xs leading-relaxed ${
                    quizChoice === activeTopic.quiz.answerIndex
                      ? "bg-emerald-50 text-emerald-900 border border-emerald-200"
                      : "bg-amber-50 text-amber-900 border border-amber-200"
                  }`}
                >
                  <strong>{quizChoice === activeTopic.quiz.answerIndex ? "Correct! " : "Not quite — "}</strong>
                  {activeTopic.quiz.explanation}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-rose-50/80 border border-rose-200/90 rounded-2xl p-5">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-rose-800 mb-1.5">
                Beginner Pitfall &amp; Debugging Tip
              </h3>
              <p className="text-xs sm:text-sm text-[#17394D] leading-relaxed">{activeTopic.pitfall}</p>
            </div>
            <div className="bg-amber-50/80 border border-amber-200/90 rounded-2xl p-5">
              <h3 className="text-xs font-extrabold uppercase tracking-wider text-amber-800 mb-1.5">
                Hands-On Practice Build
              </h3>
              <p className="text-xs sm:text-sm font-semibold text-[#17394D] leading-relaxed">{activeTopic.project}</p>
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 p-5 space-y-3">
            <h3 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
              Optional External Video &amp; Blog References
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {activeTopic.links.map((link, idx) => (
                <a
                  key={idx}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between gap-2 p-3 rounded-xl bg-slate-50 hover:bg-indigo-50/60 border border-slate-200 hover:border-indigo-300 transition-all"
                >
                  <span className="text-xs font-semibold text-[#17394D] truncate">{link.label}</span>
                  <span className="text-xs font-bold text-indigo-600 shrink-0">&#8599;</span>
                </a>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 p-4 flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              disabled={activeTopicIndex <= 0}
              onClick={() => activeTopicIndex > 0 && setActiveTopicId(ALL_TOPICS[activeTopicIndex - 1].id)}
              className="px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white text-xs sm:text-sm font-bold text-[#17394D] disabled:opacity-40"
            >
              &larr; Previous Module
            </button>

            <button
              type="button"
              onClick={() => toggleTopic(activeTopic.id)}
              className={`px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold transition-all ${
                completedIds.has(activeTopic.id)
                  ? "bg-emerald-600 text-white"
                  : "bg-[#17394D] text-white hover:bg-indigo-900"
              }`}
            >
              {completedIds.has(activeTopic.id) ? "Completed ✓" : "Mark Module Completed"}
            </button>

            <button
              type="button"
              disabled={activeTopicIndex >= ALL_TOPICS.length - 1}
              onClick={() =>
                activeTopicIndex < ALL_TOPICS.length - 1 && setActiveTopicId(ALL_TOPICS[activeTopicIndex + 1].id)
              }
              className="px-4 py-2.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-white text-xs sm:text-sm font-bold text-[#17394D] disabled:opacity-40"
            >
              Next Module &rarr;
            </button>
          </div>
        </main>
      ) : (
        /* =================================================================== */
        /* VIEW B: MAIN ROADMAP OVERVIEW + STEP 0 SETUP + 6 PHASE TIMELINE     */
        /* =================================================================== */
        <>
          <section className="relative pt-10 pb-8 sm:pt-14 sm:pb-12 border-b border-slate-200/70">
            <div className="max-w-6xl mx-auto px-4 sm:px-6">
              <div className="max-w-3xl">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-teal-200/80 shadow-xs text-xs font-semibold text-teal-800 mb-4">
                  <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping" />
                  <span>All-In-One Interactive Course &bull; 31 Full Lesson Pages Included</span>
                </div>
                <h1 className="text-3xl sm:text-5xl font-extrabold text-[#17394D] tracking-tight leading-[1.15]">
                  Build Large Language Models{" "}
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-indigo-600">
                    From Scratch
                  </span>
                </h1>
                <p className="mt-4 text-base sm:text-lg text-slate-600 leading-relaxed">
                  Click any of the 31 modules below to open its full interactive lesson page—complete with animated concept visualizers, plain-English explanations, PyTorch code walkthroughs, and self-check quizzes right inside this app.
                </p>
              </div>

              {/* Interactive Live LLM Architecture Pipeline Visualizer */}
              <div className="mt-8 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-100">
                  <div>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-indigo-500" />
                      What You Will Build: End-to-End Decoder-Only LLM Pipeline
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Click any stage in the pipeline below to inspect how raw text becomes next-token predictions—and jump to its phase.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono text-slate-500 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200/70">
                    <span>Input:</span>
                    <span className="text-indigo-600 font-semibold">&quot;The capital of France is&quot;</span>
                    <span>&rarr;</span>
                    <span className="text-teal-700 font-semibold bg-teal-50 px-1.5 py-0.5 rounded">&quot;Paris&quot;</span>
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  {PIPELINE_STAGES.map((stage, idx) => {
                    const isSelected = stage.id === activeStageId;
                    return (
                      <button
                        key={stage.id}
                        type="button"
                        onClick={() => setActiveStageId(stage.id)}
                        className={`group relative text-left p-3.5 rounded-xl border transition-all duration-200 cursor-pointer ${
                          isSelected
                            ? "bg-gradient-to-br from-teal-50/90 to-indigo-50/70 border-teal-500 shadow-xs ring-2 ring-teal-500/20"
                            : "bg-slate-50/70 hover:bg-white border-slate-200/90 hover:border-indigo-300"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span
                            className={`w-2.5 h-2.5 rounded-full ${
                              isSelected ? "bg-teal-500" : "bg-slate-300 group-hover:bg-indigo-400"
                            }`}
                          />
                          {idx < PIPELINE_STAGES.length - 1 ? (
                            <span className="hidden lg:inline text-xs font-mono text-slate-400">&rarr;</span>
                          ) : (
                            <span className="text-[10px] font-mono text-teal-700 font-bold">OUT</span>
                          )}
                        </div>
                        <div className="text-xs font-bold text-[#17394D] group-hover:text-indigo-700 transition-colors">
                          {stage.name}
                        </div>
                        <div className="text-[11px] font-mono text-slate-500 mt-0.5">{stage.sub}</div>
                      </button>
                    );
                  })}
                </div>

                <div className="mt-4 p-3.5 rounded-xl bg-gradient-to-r from-slate-50 to-indigo-50/40 border border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-all">
                  <div className="flex items-start sm:items-center gap-3">
                    <span className="shrink-0 px-2.5 py-1 rounded-md text-xs font-mono font-semibold bg-indigo-100 text-indigo-800">
                      {activeStage.badge}
                    </span>
                    <p className="text-xs sm:text-sm text-[#17394D] font-medium">{activeStage.desc}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleJumpToPhase(activeStage.phaseId)}
                    className="shrink-0 text-xs font-semibold text-teal-700 hover:text-teal-900 bg-white hover:bg-teal-50 border border-teal-200 px-3 py-1.5 rounded-lg shadow-2xs transition-all self-start sm:self-auto"
                  >
                    Jump to Phase &darr;
                  </button>
                </div>
              </div>

              {/* Step 0: Environment Setup */}
              <div className="mt-6 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 sm:p-6 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100">
                  <div>
                    <h2 className="text-sm font-bold uppercase tracking-wider text-[#17394D] flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-teal-500" />
                      Step 0: Required Softwares, Python Packages &amp; Setup Commands
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Install only what you need for building and running LLMs in Python &amp; PyTorch (no Java/JDK/Maven needed).
                    </p>
                  </div>
                  <a
                    href="https://pytorch.org/get-started/locally/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="shrink-0 inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-700 hover:text-indigo-900 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 px-3 py-1.5 rounded-lg transition-colors self-start sm:self-auto"
                  >
                    <span>PyTorch GPU/CPU Selector</span>
                    <span>&#8599;</span>
                  </a>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {SETUP_SOFTWARES.map((sw) => (
                    <a
                      key={sw.name}
                      href={sw.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group p-3.5 rounded-xl bg-slate-50/70 hover:bg-white border border-slate-200/90 hover:border-teal-400 transition-all flex flex-col justify-between gap-2.5"
                    >
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-bold text-[#17394D] group-hover:text-teal-700">{sw.name}</span>
                          <span className="text-[11px] font-bold text-teal-700 shrink-0">Download &#8599;</span>
                        </div>
                        <p className="text-[11px] font-medium text-slate-700 leading-snug">{sw.role}</p>
                        <div className="p-2 rounded-lg bg-violet-50/75 border border-violet-200/75 text-[10px] text-[#17394D] leading-snug">
                          <span className="font-bold text-violet-900">Where Used: </span>
                          {sw.whereUsed}
                        </div>
                        <div className="p-2 rounded-lg bg-emerald-50/70 border border-emerald-200/75 text-[10px] text-[#17394D] leading-snug">
                          <span className="font-bold text-emerald-900">What You Build: </span>
                          {sw.whatYouBuild}
                        </div>
                      </div>
                      <code className="block text-[10px] font-mono bg-white border border-slate-200/80 rounded px-2 py-1 text-slate-600 truncate">
                        {sw.winget}
                      </code>
                    </a>
                  ))}
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl bg-sky-50/60 border border-sky-200/80 p-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                      <div>
                        <span className="text-[11px] font-extrabold uppercase tracking-wider text-sky-950 block">
                          14 Python Packages (PyPI) — Purpose, Live AI Usage &amp; What You Will Build
                        </span>
                      </div>
                      <a
                        href="https://pypi.org/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] font-semibold text-sky-800 hover:underline shrink-0"
                      >
                        pypi.org &#8599;
                      </a>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                      {SETUP_PACKAGES.map((pkg) => (
                        <a
                          key={pkg.name}
                          href={`https://pypi.org/project/${pkg.name}/`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="group bg-white hover:bg-teal-50/40 border border-sky-200/90 hover:border-teal-400 rounded-xl p-3 text-[#17394D] transition-all flex flex-col justify-between gap-1.5"
                        >
                          <div>
                            <div className="flex items-center justify-between gap-2">
                              <strong className="font-mono text-xs text-indigo-700 group-hover:text-teal-700">
                                {pkg.name}
                              </strong>
                              <span className="text-[10px] font-semibold text-slate-400 group-hover:text-teal-700">
                                PyPI &#8599;
                              </span>
                            </div>
                            <p className="text-[11px] font-semibold text-[#17394D] mt-0.5 leading-snug">{pkg.desc}</p>
                          </div>
                          <div className="space-y-1 pt-1.5 border-t border-slate-100 text-[10px] leading-snug">
                            <p className="text-slate-600">
                              <strong className="text-violet-900">Where used:</strong> {pkg.whereUsed}
                            </p>
                            <p className="text-slate-700">
                              <strong className="text-emerald-900">You build:</strong> {pkg.whatYouBuild}
                            </p>
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-xl bg-slate-900 border border-slate-800 p-4 flex flex-col justify-between">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span className="text-[11px] font-mono font-semibold text-teal-400">
                        Windows PowerShell Setup &amp; Verification Commands
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          navigator.clipboard?.writeText(SETUP_COMMANDS_TEXT);
                          setCopiedSetup(true);
                          setTimeout(() => setCopiedSetup(false), 2200);
                        }}
                        className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                      >
                        {copiedSetup ? "Copied!" : "Copy Commands"}
                      </button>
                    </div>
                    <pre className="text-[11px] font-mono text-slate-100 overflow-x-auto leading-relaxed">
                      <code>{SETUP_COMMANDS_TEXT}</code>
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Filter Bar & Controls */}
          <section className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="Roadmap Phase Filters">
                {[
                  { id: "all", label: "All Phases (6)" },
                  { id: "phase-1", label: "Phase 1: Foundations" },
                  { id: "phase-2", label: "Phase 2: Autograd" },
                  { id: "phase-3", label: "Phase 3: Tokenization" },
                  { id: "phase-4", label: "Phase 4: Transformer" },
                  { id: "phase-5", label: "Phase 5: Pretraining" },
                  { id: "phase-6", label: "Phase 6: Fine-Tune & Serve" }
                ].map((tab) => {
                  const isMatch = currentFilter === tab.id;
                  return (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setCurrentFilter(tab.id)}
                      className={`px-3 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
                        isMatch
                          ? "bg-[#17394D] text-white shadow-xs"
                          : "bg-white text-slate-600 hover:text-[#17394D] border border-slate-200/90 hover:border-slate-300"
                      }`}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center gap-2 self-start md:self-auto">
                <button
                  type="button"
                  onClick={() => setCollapsedPhases(new Set())}
                  className="text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
                >
                  Expand All
                </button>
                <button
                  type="button"
                  onClick={() => setCollapsedPhases(new Set(ROADMAP_PHASES.map((p) => p.id)))}
                  className="text-xs font-semibold text-slate-600 hover:text-[#17394D] bg-white border border-slate-200 px-3 py-2 rounded-xl hover:bg-slate-50 transition"
                >
                  Collapse All
                </button>
              </div>
            </div>
          </section>

          {/* Main Timeline Roadmap */}
          <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 pb-24">
            <div className="relative" ref={timelineRef}>
              <div className="hidden md:block absolute left-7 top-4 bottom-8 w-1 bg-slate-200/90 rounded-full overflow-hidden">
                <div
                  className="w-full bg-gradient-to-b from-teal-500 via-indigo-500 to-amber-500 rounded-full transition-all duration-150 ease-out"
                  style={{ height: `${scrollSpinePct}%` }}
                />
              </div>

              <div className="space-y-8">
                {filteredPhases.map((phase) => {
                  const theme = getAccentClasses(phase.accent);
                  const phaseDoneCount = phase.topics.filter((t) => completedIds.has(t.id)).length;
                  const phaseTotal = phase.topics.length;
                  const pct = Math.round((phaseDoneCount / phaseTotal) * 100);
                  const isPhaseComplete = phaseDoneCount === phaseTotal;
                  const isCollapsed = collapsedPhases.has(phase.id);

                  return (
                    <article key={phase.id} id={phase.id} className="relative md:pl-20">
                      <div
                        className={`hidden md:flex absolute left-0 top-5 w-14 h-14 rounded-2xl border-2 ${
                          isPhaseComplete ? "border-emerald-500 bg-emerald-500 text-white" : theme.nodeRing
                        } shadow-sm items-center justify-center font-mono font-extrabold text-base z-10 transition-colors`}
                      >
                        {isPhaseComplete ? "✓" : phase.number}
                      </div>

                      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden">
                        <div
                          onClick={() => togglePhaseCollapse(phase.id)}
                          className="p-5 sm:p-6 cursor-pointer select-none flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors"
                        >
                          <div className="space-y-1.5 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="md:hidden font-mono text-xs font-bold px-2 py-0.5 rounded-md bg-[#17394D] text-white">
                                Phase {phase.number}
                              </span>
                              <span className={`text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full border ${theme.badgeBg}`}>
                                {phase.duration}
                              </span>
                              <span className="text-xs font-medium text-slate-400">
                                &bull; {phaseDoneCount}/{phaseTotal} modules completed
                              </span>
                            </div>
                            <h2 className="text-lg sm:text-xl font-extrabold text-[#17394D] tracking-tight">{phase.title}</h2>
                            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">{phase.subtitle}</p>

                            <div className="mt-2.5 p-3 rounded-xl bg-sky-50/75 border border-sky-200/80 text-xs text-[#17394D] leading-relaxed">
                              <span className="font-bold text-sky-900 uppercase tracking-wider text-[10px] mr-1.5">
                                Plain-English Big Picture:
                              </span>
                              {phase.plainBigPicture}
                            </div>

                            {phase.liveAiBridge && (
                              <div className="mt-2.5 p-3.5 rounded-xl bg-gradient-to-r from-violet-50/85 via-indigo-50/60 to-white border border-violet-200/85 text-xs text-[#17394D] leading-relaxed">
                                <div className="flex items-center gap-1.5 mb-1">
                                  <span className="w-2 h-2 rounded-full bg-violet-600 animate-pulse" />
                                  <span className="font-extrabold text-violet-950 uppercase tracking-wider text-[10px]">
                                    {phase.liveAiBridge.headline}
                                  </span>
                                </div>
                                <p className="text-slate-700 mb-2">{phase.liveAiBridge.summary}</p>
                                <ul className="space-y-1">
                                  {phase.liveAiBridge.examples.map((ex, idx) => (
                                    <li key={idx} className="flex items-start gap-1.5 text-[11px] text-[#17394D]">
                                      <span className="text-violet-600 font-bold shrink-0 mt-px">&rarr;</span>
                                      <span>{ex}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            <div className="mt-2 flex flex-wrap items-center gap-1.5 pt-0.5">
                              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mr-1">
                                Jargon Decoder:
                              </span>
                              {phase.jargonDecoder.map((item) => (
                                <span
                                  key={item.term}
                                  className="inline-flex items-center gap-1 text-[11px] bg-slate-50 border border-slate-200/90 rounded-lg px-2 py-0.5 text-slate-700"
                                >
                                  <strong className="font-mono text-indigo-700">{item.term}</strong>
                                  <span className="text-slate-400">=</span>
                                  <span>{item.def}</span>
                                </span>
                              ))}
                            </div>
                          </div>

                          <div className="flex items-center justify-between sm:justify-end gap-4 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100">
                            <div className="w-28">
                              <div className="flex justify-between text-[11px] font-mono font-semibold text-slate-500 mb-1">
                                <span>Progress</span>
                                <span>{pct}%</span>
                              </div>
                              <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${isPhaseComplete ? "bg-emerald-500" : theme.bar} rounded-full transition-all duration-500`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                            <button
                              type="button"
                              aria-label="Toggle phase details"
                              className={`w-8 h-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center transition-transform duration-300 ${
                                isCollapsed ? "rotate-180" : ""
                              }`}
                            >
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                              </svg>
                            </button>
                          </div>
                        </div>

                        {!isCollapsed && (
                          <div className="px-5 sm:px-6 pb-6 pt-2 border-t border-slate-100 space-y-4">
                            <div className="mt-2 rounded-xl bg-gradient-to-r from-indigo-50/80 via-sky-50/45 to-white border border-indigo-200/80 p-4">
                              <div className="flex items-center gap-2 mb-2.5">
                                <span className="text-[11px] font-extrabold uppercase tracking-wider text-indigo-950">
                                  What You Will Learn in Phase {phase.number} — Click Any Module to Open Its Interactive Lesson Page
                                </span>
                              </div>
                              <div className="flex flex-wrap gap-2">
                                {phase.topics.map((t) => (
                                  <button
                                    key={t.id}
                                    type="button"
                                    onClick={() => setActiveTopicId(t.id)}
                                    className="inline-flex items-center gap-1.5 text-left bg-white/90 hover:bg-indigo-50/80 border border-indigo-200/75 hover:border-indigo-400 rounded-lg px-2.5 py-1.5 text-xs text-[#17394D] font-semibold transition-all shadow-2xs"
                                  >
                                    <span className="font-mono text-[11px] font-bold text-indigo-700">{t.codeNum}</span>
                                    <span>{t.title}</span>
                                  </button>
                                ))}
                              </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                              {phase.topics.map((topic) => {
                                const checked = completedIds.has(topic.id);
                                return (
                                  <div
                                    key={topic.id}
                                    onClick={() => setActiveTopicId(topic.id)}
                                    className={`group relative flex items-start gap-3 p-4 rounded-xl border transition-all duration-200 cursor-pointer ${
                                      checked
                                        ? "bg-emerald-50/45 border-emerald-200/90"
                                        : `bg-slate-50/60 border-slate-200/80 ${theme.pillHover} hover:bg-white hover:shadow-xs`
                                    }`}
                                  >
                                    <div className="pt-0.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={() => toggleTopic(topic.id)}
                                        aria-label={`Mark ${topic.title} completed`}
                                        className="w-4 h-4 rounded text-teal-600 focus:ring-teal-500 border-slate-300 cursor-pointer"
                                      />
                                    </div>

                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center justify-between gap-2">
                                        <div className="flex items-center gap-1.5">
                                          <span className="text-[11px] font-mono font-bold px-1.5 py-0.5 rounded bg-white border border-slate-200/90 text-[#17394D]">
                                            {topic.codeNum}
                                          </span>
                                          <span className="text-[11px] font-mono text-slate-400">{topic.duration}</span>
                                        </div>
                                        <span className="text-[11px] font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-md group-hover:bg-indigo-600 group-hover:text-white transition-all shrink-0">
                                          Open Lesson Page &rarr;
                                        </span>
                                      </div>

                                      <h3
                                        className={`text-xs sm:text-sm font-bold leading-snug mt-1.5 ${
                                          checked
                                            ? "text-slate-500 line-through decoration-emerald-400"
                                            : "text-[#17394D] group-hover:text-indigo-950"
                                        }`}
                                      >
                                        {topic.title}
                                      </h3>

                                      <div
                                        className={`mt-2 p-2.5 rounded-lg bg-sky-50/70 border border-sky-200/70 text-[11px] leading-relaxed ${
                                          checked ? "text-slate-400" : "text-[#17394D]"
                                        }`}
                                      >
                                        <span className="font-bold text-sky-900">What is this? </span>
                                        {topic.plainEnglish}
                                      </div>

                                      {topic.liveExample && (
                                        <div
                                          className={`mt-2 p-2.5 rounded-lg bg-violet-50/75 border border-violet-200/80 text-[11px] leading-relaxed ${
                                            checked ? "text-slate-400" : "text-[#17394D]"
                                          }`}
                                        >
                                          <span className="font-bold text-violet-900">Live AI In Action: </span>
                                          {topic.liveExample}
                                        </div>
                                      )}

                                      <ul className="mt-2.5 space-y-1 border-t border-slate-200/60 pt-2">
                                        {topic.concepts.map((concept, idx) => (
                                          <li
                                            key={idx}
                                            className={`flex items-start gap-1.5 text-[11px] leading-snug ${
                                              checked ? "text-slate-400" : "text-slate-600"
                                            }`}
                                          >
                                            <span className="text-teal-600 font-bold shrink-0 mt-px">&bull;</span>
                                            <span>{concept}</span>
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>

                            <div className="rounded-xl bg-gradient-to-r from-emerald-50/80 via-teal-50/45 to-white border border-emerald-200/85 p-4">
                              <div className="text-[11px] font-extrabold uppercase tracking-wider text-emerald-900 mb-2.5">
                                What You Will Achieve &amp; Master After Phase {phase.number}
                              </div>
                              <ul className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                                {phase.achievements.map((item, idx) => (
                                  <li
                                    key={idx}
                                    className="flex items-start gap-2 bg-white/85 rounded-lg p-2.5 border border-emerald-200/60 text-xs text-[#17394D] leading-snug"
                                  >
                                    <span className="text-emerald-600 font-extrabold shrink-0 mt-px">&#10003;</span>
                                    <span>{item}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>

                            <div className="rounded-xl bg-gradient-to-r from-amber-50/80 via-orange-50/40 to-white border border-amber-200/80 p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                              <div>
                                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800 block">
                                  Phase {phase.number} Milestone Build
                                </span>
                                <span className="text-xs sm:text-sm font-bold text-[#17394D]">{phase.miniProject}</span>
                              </div>
                              <button
                                type="button"
                                onClick={() => setActiveTopicId(phase.topics[phase.topics.length - 1].id)}
                                className="text-xs font-semibold text-amber-900 hover:text-amber-700 underline underline-offset-2 self-start sm:self-center shrink-0"
                              >
                                Open Milestone Lesson &rarr;
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>

            <div className="mt-12 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
              <div>
                <span className="inline-block text-xs font-bold uppercase tracking-wider text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md mb-2">
                  Golden Rule for Beginners
                </span>
                <h3 className="text-xl font-bold text-[#17394D]">
                  Don&apos;t just watch tutorials—print tensor shapes at every step!
                </h3>
                <p className="text-sm text-slate-600 mt-1.5 max-w-2xl">
                  90% of bugs when building LLMs from scratch come from mismatched matrix dimensions{" "}
                  <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded text-indigo-700">
                    (Batch, Seq_Len, Embed_Dim)
                  </code>
                  . Start with Module 1.1&apos;s interactive lesson page below and build every layer yourself.
                </p>
              </div>
              <div className="shrink-0 flex flex-col sm:flex-row gap-3 w-full md:w-auto">
                <button
                  type="button"
                  onClick={() => setActiveTopicId(ALL_TOPICS[0].id)}
                  className="px-5 py-3 rounded-xl bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 text-white font-semibold text-sm shadow-sm transition-all text-center"
                >
                  Open Module 1.1 Lesson Page &rarr;
                </button>
              </div>
            </div>
          </main>
        </>
      )}

      {/* Non-Blocking Toast Banner */}
      {toast && (
        <div className="fixed bottom-5 right-5 z-50 bg-[#17394D] text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3 text-xs sm:text-sm">
          <span>{toast.message}</span>
          {toast.confirm && (
            <button
              type="button"
              onClick={() => {
                setCompletedIds(new Set());
                setToast({ message: "Roadmap progress cleared.", confirm: false });
                setTimeout(() => setToast(null), 2500);
              }}
              className="px-2.5 py-1 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-semibold text-xs"
            >
              Yes, Reset
            </button>
          )}
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-slate-300 hover:text-white font-semibold text-xs"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}