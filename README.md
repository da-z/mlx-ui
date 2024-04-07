# MLX Chat

A simple UI / Web / Frontend for MLX mlx-lm using Streamlit.

![](screenshot.png)

## Install

```shell
$ ./install.sh
```

Or to install using latest versions of the libs (may break functionality):

```shell
$ ./install.sh refresh
```

## Update

After fetching a newer version it's recommended to run again the installation script.

```shell
$ ./install.sh
```

## Run

```shell
$ ./run.sh
```

You can also use a custom model.txt file (see [mlx-community](https://huggingface.co/mlx-community) for more models):

```shell
$ ./run.sh --models mymodels.txt
```
