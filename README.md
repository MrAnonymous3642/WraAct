# WraAct: Convex Hull Approximation for Activation Functions

This is an artifact for the tool called **WraAct** to over-approximate the function hulls of various activation functions (including leaky relu, relu, sigmoid, tanh, and maxpool).

This is for the paper **Convex Hull Approximation for Activation Functions**.

# Installation

## Requirements and Dependencies

The following only demonstrate the installation of our tool rather than all the baseline approaches.

### Python Version

Our code is pure [Python](https://www.python.org/), but you need a version of >=3.10 (we are using 3.12) to support some functions on [typing](https://docs.python.org/3/library/typing.html). Also, you need install the following libraries on a Linux system.

### Python Environment

Consider the dependencies of libraries, we list the following libraries in an order of our installation. Actually, some other versions of the following libraries maybe also fine because our code does not need special methods.

We install the tool on an [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.anaconda.com/miniconda/) environment by `pip` or `conda` (some library installations do not support `conda`). Please install [Anaconda](https://www.anaconda.com/)/[Miniconda](https://docs.anaconda.com/miniconda/) first.

First, we create an environment and activate it.

```cmd
conda create --name wraact python=3.12
conda activate wraact
```

### Python Libraries

> The following libraries can use the higher versions in most cases, but we list the versions we used in our experiments.

You can install a GPU-version by [pytorch installation](https://pytorch.org/get-started/previous-versions/). We use the following versions but the higher versions of PyTorch and CUDA are also fine. Note that [NumPy](https://numpy.org/) will be installed after installing [PyTorch](https://pytorch.org/).

- pytorch==2.3.1
- torchvision==0.18.1
- torchaudio==2.3.1
- pytorch-cuda=12.1

Then, you need the library for programming solver Gurobi and install it by `pip` (not support `conda`). To use Gurobi, you need a license and [find a license right for you](https://www.gurobi.com/academia/academic-program-and-licenses/).

```cmd
pip install gurobipy==11.0.3
```

Also, you need the following libraries, where `scipy` is for extracting sparse constraints matrix form `gurobipy`, `onnx` is for reading ONNX format models, `onnxruntime` is for checking the correctness of ONNX models, and `numba` is for compiling some functions (about tangent lines) to speed up the code.

```cmd
pip install scipy==1.15.3 onnx==1.18.0 onnxruntime==1.22.0 numba==0.61.2
```

The following library _needs a specified version_ because there are some changes in the latest version of `pycddlib` that will cause the code to fail. We use the version `2.1.7` in our experiments. `pycddlib` is for calculating the vertices of a convex polytope in this work. It is friendly to `numpy` operations.

- pycddlib==2.1.7

```cmd
pip install pycddlib==2.1.7
```

You also need to install `matplotlib` if you want to produce our figures in the paper. You also need TexLive if you want to plot LaTeX fonts (This is not necessary). You need to install [ELINA](https://github.com/eth-sri/ELINA) if you want to use the method in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) called SBLM+PDDM to compare it with our method. You can leave it when you run the command in the following sections.

Currently, you have installed all required libraries.

# Main Code Structure

The folder structure of this repository is as follows. We only list the main folders and files here. The source code of WraAct is in the `src/` folder, which contains the code for bound propagation, function hull approximation, linear programming, model building, and utility functions. The evaluation code is in the `evaluation_volume` and `evaluation_verification` folders. The archieved logs of our paper are in the `archieved_logs` folder. The benchmark models are in the `nets` folder. The other folders are for baseline approaches, including ELINA for SBLM+PDDM in PRIMA, ERAN for DeepPoly and PRIMA, and auto-LiRPA for vanilla CROWN.

```
WraAct/                      # Main folder of WraAct
├── .temp/                   # Temporary files (e.g., downloaded datasets)
├── archieved_logs/          # Archieved logs of our paper
├── evaluation_volume/       # Code for volume evaluation of convex hull approximation
├── evaluation_verification/ # Code for local robustness verification
├── nets/                    # Benchmark ONNX models
├── src/                     # Source code of WraAct
│   ├── boundprop/           # Code for bound propagation
│   │   ├── ineq/            # Linear inequalities bounds for propagation
│   │   │   ├── backsub/     # Symbolic back-substitution for inequalities
│   │   │   └── relaxation/  # Linear relaxation for activation functions
│   │   └── base.py          # Base classes for propagation
│   ├── funchull/            # Code for function hull approximation
│   ├── linprog/             # Code for linear programming
│   ├── model/               # Code for verification models
│   └── utils/               # Miscellaneous utility functions
└── README.md
ELINA/                       # ELINA project for SBLM+PDDM in PRIMA; for baseline comparison
ERAN/                        # ERAN project for DeepPoly and PRIMA; for baseline comparison
auto-LiRPA/                  # auto_LiRPA project for vanilla CROWN; for baseline comparison
```

We have a good class structure for different activation functions and layer structure of neural networks. If you want to extend the code to support new functions or layers, you need to modify the code in the corresponding folder in `src/`.

# Usage

This section describes how to run the evaluation code in the paper. The evaluation code is in the `evaluation_volume` and `evaluation_verification` folders.

> TIP: You can download our original logs and benchmark onnx models from the following Google drive [**sharing link**](https://drive.google.com/drive/folders/1C4kYaKb_Pd3xCo6aCy6W80tw43CM8Nn8?usp=sharing).

## Evaluation: Function Hull Approximation

The `evaluation_volume` folder contains the code about volume evaluation of convex hull approximation for activation functions. To evaluate the convex hull approximation method in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704), you need install [ELINA](https://github.com/eth-sri/ELINA) and put it in the `WraAct` folder if you need test the methods in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) called SBLM+PDDM. If you only want to use WraAct, you do not need to install ELINA.

### Check Achieved Results

The subfolder `archieved_logs/evaluation_volume` contains the experiment logs of our paper.

### Reproduce Results

There are 4 steps in 4 folders for different purposes. To run the following commands, you need to make sure you are in the corresponding subfolder.

**Step 1: Generate Polytope Samples**

Folder `polytope_samples`: This is for generating samples of convex polytopes. This folder will contain several `.txt` files to record the generated samples after you run the following command. This process can be completed within 10s in our machine.

```cmd
python3 generate_polytope_samples.py
```

> TIP: Because SBLM+PDDM only support octahedrons as input polytopes, we will generate octahedrons for it based on the original random input polytopes. Also, you will see SBLM+PDDM will only calculate these octahedrons in the following steps.

**Step 2: Prepare Polytope Bounds**

Folder `polytope_bounds`: This is for calculate the bounds of each dimension of the given convex polytopes. It is used for the algorithm in [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704) called SBLM+PDDM. This folder will also contain several `.txt` files to record the bounds of each dimension after you run the following command. This process can be completed within 2m in our machine.

```cmd
python3 calculate_polytope_bounds.py
```

**Step 3: Calculate Function Hulls**

Folder `output_constraints`: This is for calculating the function hull approximation of activation functions. This folder will also contain several `.txt` files to record the constraints of the convex hull approximation after you run the following command. This process can be completed within 6m in our machine.

```cmd
python3 calculate_function_hulls.py
```

> WARNING: When you run `output_constraints.py` it need to call the functions in [ELINA](https://github.com/eth-sri/ELINA) for SBLM+PDDM in the paper [PRIMA](https://dl.acm.org/doi/pdf/10.1145/3498704). If you do not install [ELINA](https://github.com/eth-sri/ELINA), you will have some warnings.``

When you have the data files in the `output_constraints` folder, you can run the following command to organize the data and plot the line chart in our paper. Sometimes, there will be some warnings about the lost data but it will not affect the results because SBLM+PDDM does not support high-dimensional polytopes. When plotting the line chart, we have disabled the latex font in the figure to avoid some errors. This just for archiving the results in our paper. If you want to plot the latex font in the figure, you can uncomment the lines in the head of the script `plot_line_chart.py`.
This process can be completed within 1s.

```cmd
python3 organize_data.py
python3 plot_line_chart.py
```

**Step 4: Calculate Volumes of Function Hulls**

Folder `hull_volumes`: This is to estimate the volume of the convex hull approximation. This folder will contain several `.txt` files to record the estimated volumes after you run the following code.
This process is time consuming and is completed in 10m in our machine.

```cmd
python3 estimate_volumes.py
```

Then, you can run the following command to output the data table in our paper. The data will be printed in the terminal.

```cmd
python3 output_data_table.py
```

**Run All Steps**

If you want to run all the steps in one command. You can run the following bash script in the `evaluation_volume` folder.

```cmd
bash evaluate_volume.sh
bash output_data.sh
```

## Evaluation: Local Robustness Verification

The `evalueation_verification` folder contains the evaluation code about local robustness verification.

### Benchmark MNIST/CIFAR10 Datasets

The dataset MNIST and CIFAR-10 will be downloaded automatically when you run the code by PyTorch. And it is saved in the `WraAct/.temp` folder.

### Benchmark ONNX Models

The benchmark models are saved in `WraAct/nets` folder. Because the file size is large, we do not upload them to the repository. You can download them from the [link]()

### About Other Baseline Approaches

For DeepPoly and PRIMA, you can install them from [ERAN](https://github.com/eth-sri/eran).
For vanilla CROWN, You can install it from [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA).

### Example to Run

There is an example script `exp_test.py` to run a small instance of local robustness verification. You can run this python script by the following command `bash test.sh` in the `evaluation_verification` folder to check if the tool works well.

```cmd
bash test.sh
```

### Check Achieved Results

The folder `achieved_logs/evaluation_verification` contains the logs of local robustness verification in our paper.

### Reproduce Results

All the evaluation scripts call `exp.py` for running experiments. You can see the following bash files in the `evaluation_folder`:

- `mnist_sshape.sh`: for S-shape functions (sigmoid, tanh) on MNIST dataset
- `mnist_relulike.sh`: for relu-like functions (elu, leaky relu) on MNIST dataset
- `cifar10_sshape.sh`: for S-shape functions (sigmoid, tanh) on CIFAR-10 dataset
- `cifar10_relulike.sh`: for relu-like functions (elu, leaky relu) on CIFAR-10 dataset
- `cifar10_maxpool.sh`: for maxpool on MNIST and CIFAR-10 dataset
- `resnet.sh`: for ResNet benchmark on CIFAR-10 dataset

Run the above bash files and you can collect the log files in the `evaluation_verification/logs` folder.

> ATTENTION: Run all the codes takes a long time, so we suggest you run them one by one. You can also comment some lines in the bash files to reduce the number of experiments.

```cmd
bash mnist_sshape.sh
bash mnist_relulike.sh
bash cifar10_sshape.sh
bash cifar10_relulike.sh
bash cifar10_maxpool.sh
bash resnet.sh
```