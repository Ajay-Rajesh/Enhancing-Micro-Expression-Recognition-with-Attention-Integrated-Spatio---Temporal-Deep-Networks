<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micro-Expression 3D CNN Framework Documentation</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --border-color: #30363d;
            --container-bg: #161b22;
            --accent-color: #58a6ff;
            --accent-bg: rgba(56, 139, 253, 0.15);
            --code-bg: #1f242c;
            --table-header: #21262d;
            --table-row-alt: #1c2128;
            --success-color: #238636;
            --warning-color: #d29922;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1012px;
            margin: 0 auto;
            background-color: var(--container-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 2.5rem;
        }

        h1, h2, h3 {
            color: #f0f6fc;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.3em;
        }

        h1 {
            font-size: 2.25rem;
            border-bottom: 1px solid var(--border-color);
            margin-top: 0;
        }

        h2 {
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        h3 {
            font-size: 1.25rem;
        }

        p {
            margin-top: 0;
            margin-bottom: 16px;
        }

        ul {
            padding-left: 2em;
            margin-top: 0;
            margin-bottom: 16px;
        }

        li {
            margin-top: 0.25em;
        }

        a {
            color: var(--accent-color);
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        code {
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 85%;
            background-color: rgba(110, 118, 129, 0.4);
            padding: .2em .4em;
            border-radius: 6px;
            color: #f0f6fc;
        }

        pre {
            background-color: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            margin-top: 0;
            margin-bottom: 16px;
        }

        pre code {
            background-color: transparent;
            padding: 0;
            font-size: 100%;
            color: #e6edf3;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0;
            margin-bottom: 16px;
        }

        th, td {
            padding: 6px 13px;
            border: 1px solid var(--border-color);
        }

        th {
            font-weight: 600;
            background-color: var(--table-header);
            text-align: left;
        }

        tr:nth-child(even) {
            background-color: var(--table-row-alt);
        }

        .alert {
            background-color: var(--accent-bg);
            border-left: .25em solid var(--accent-color);
            padding: 16px;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
            margin-bottom: 16px;
        }

        .alert-title {
            font-weight: 600;
            margin-bottom: 4px;
            color: var(--accent-color);
        }

        .badge {
            display: inline-block;
            padding: 2px 7px;
            font-size: 12px;
            font-weight: 500;
            line-height: 18px;
            border-radius: 2em;
            background-color: var(--border-color);
            color: var(--text-muted);
            margin-right: 4px;
        }

        .badge-cuda { background-color: var(--success-color); color: #fff; }
        .badge-model { background-color: #5319e7; color: #fff; }

        .architecture-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .arch-card {
            border: 1px solid var(--border-color);
            background-color: var(--table-row-alt);
            border-radius: 6px;
            padding: 16px;
        }

        .arch-card h3 {
            margin-top: 0;
            color: var(--accent-color);
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Micro-Expression 3D CNN Framework</h1>
    <p>
        <span class="badge badge-cuda">PyTorch 2.0+</span>
        <span class="badge badge-cuda">CUDA Accelerated</span>
        <span class="badge badge-model">3D-CNN</span>
        <span class="badge badge-model">R3D-18</span>
    </p>
    <p>A comprehensive PyTorch framework for video-based micro-expression recognition using state-of-the-art spatial-temporal deep learning models. This repository implements custom video sequence datasets, robust frame-handling data loaders, and three distinct neural network architectures optimized for micro-expression classification.</p>

    <h2>🚀 Key Framework Features</h2>
    <ul>
        <li><strong>Spatial-Temporal Backends:</strong> Implements Vanilla 3D Convolution networks, Advanced 3D ResNet Transfer Learning, and custom Squeeze-and-Excitation 3D Attention blocks.</li>
        <li><strong>Deterministic Sequence Engineering:</strong> Formulates robust frame handling. Employs dynamic uniform interval sampling for long videos and terminal frame padding duplicates for short clips.</li>
        <li><strong>Mixed Precision Training:</strong> Integrates modern <code>torch.amp</code> automatic mixed-precision context blocks, yielding major training speedups and lower VRAM usage on modern GPUs.</li>
        <li><strong>Production Optimization:</strong> Leverages adaptive pooling mapping, gradient clipping protection, automated class-imbalance weight balancing calculation, and early stopping criteria.</li>
    </ul>

    <h2>🏗️ Architecture Catalog</h2>
    <div class="architecture-grid">
        <div class="arch-card">
            <h3>1. Simple 3D CNN</h3>
            <p>A standard sequential spatial-temporal network featuring progressive convolutional channels (3 &rarr; 32 &rarr; 64 &rarr; 128) paired with 3D Batch Normalization, ReLU activations, and MaxPool3D filters.</p>
        </div>
        <div class="arch-card">
            <h3>2. Advanced R3D-18</h3>
            <p>Utilizes a pretrained <code>torchvision.models.video.r3d_18</code> base layer. Extracted backend features pass into a custom dropout classification head optimized with layer-specific differential learning rates.</p>
        </div>
        <div class="arch-card">
            <h3>3. Model B with Channel Attention</h3>
            <p>A custom deep-learning model engineering Squeeze-and-Excitation (SE) channel dependencies. Standardized feature channels are weighted using Sigmoid excitation parameters generated from spatial-temporal adaptive average pools.</p>
        </div>
    </div>

    <h2>📂 Project Directory Structure</h2>
    <pre><code>📂 Micro-Expression-3D-CNN
├── 📂 data                    # Data directory
│   └── 📂 processed_split     # Subject-independent dataset splits
│       ├── 📂 train           # Class-stratified training video folders
│       ├── 📂 val             # Validation verification folders 
│       └── 📂 test            # Final model generalization test set
├── 📄 dataset.py              # Sequence loading utilities and dynamic frame engines
├── 📄 models.py               # Deep network architectural design declarations
├── 📄 train.py                # Native AMP training pipelines and checkpoint managers
└── 📄 README.html             # Repository HTML documentation</code></pre>

    <h3>Expected Layout of Class Folder Sequence</h3>
    <pre><code>📂 processed_32_subject_split/
├── 📂 train/
│   ├── 📂 class_00_positive/
│   │   ├── 📂 video_sample_01/
│   │   │   ├── 📄 frame_01.jpg
│   │   │   └── 📄 frame_02.jpg
│   │   └── 📂 video_sample_02/
│   └── 📂 class_01_negative/
└── 📂 val/</code></pre>

    <h2>⚙️ Core Framework Hyperparameters</h2>
    <table>
        <thead>
            <tr>
                <th>Configuration Key</th>
                <th>Default Configuration Value</th>
                <th>Operational Engineering Intent</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>IMG_SIZE</code></td>
                <td><code>(128, 128)</code></td>
                <td>Spatial height/width configurations applied to incoming frames.</td>
            </tr>
            <tr>
                <td><code>FRAMES</code></td>
                <td><code>32</code></td>
                <td>Target frame sequence length enforced across all video tensor shapes.</td>
            </tr>
            <tr>
                <td><code>BATCH_SIZE</code></td>
                <td><code>6</code></td>
                <td>VRAM bounding constraint managing high-dimensional 5D network inputs.</td>
            </tr>
            <tr>
                <td><code>LR</code></td>
                <td><code>1e-4</code></td>
                <td>Baseline weight learning rate velocity optimizing the neural weights.</td>
            </tr>
            <tr>
                <td><code>WEIGHT_DECAY</code></td>
