# dmc-2026-code
Code example to the DMC 2026 Paper: 

**From Optimization to Reasoning: An Agentic Approach to Component Placement for PCB Layout**


## KiCAD Installation

[KiCAD](https://www.kicad.org/) and the [KiCAD MCP](https://github.com/mixelpixx/KiCAD-MCP-Server) server are running inside a podman container on a Linux host system.

**Container setup:**

```dockerfile
FROM fedora:42

RUN dnf install -y 'dnf-command(copr)' && \
    dnf copr enable -y @kicad/kicad-stable && \
    dnf install -y \
      kicad \
      python3-pip \
      nodejs \
      npm \
      git \
      which \
      cairo-gobject \
      python3-cairo && \
    dnf clean all

RUN python3 -c "import pcbnew; print(pcbnew.GetBuildVersion())"

ENV PYTHONPATH=/usr/lib64/python3.13/site-packages

RUN pip3 install \
      mcp \
      openai \
      kiutils \
      pdfplumber \
      sexpdata>=1.0.2 \
      kicad-skip>=0.1.0 \
      Pillow>=9.0.0 \
      cairosvg>=2.7.0 \
      colorlog>=6.7.0 \
      pydantic>=2.5.0 \
      requests>=2.32.5 \
      python-dotenv>=1.0.0

RUN git clone https://github.com/mixelpixx/KiCAD-MCP-Server.git \
      /workspace/KiCAD-MCP-Server && \
    cd /workspace/KiCAD-MCP-Server && \
    npm install && \
    npm run build

WORKDIR /workspace
```

**Building the container**

```sh
podman build -t kicad-mcp .
```

**Opening the KiCAD GUI inside the container**

```sh
xhost +local:podman 

podman run --rm -it \ -e DISPLAY=$DISPLAY \ -v /tmp/.X11-unix:/tmp/.X11-unix:Z \ -v $HOME/kicad_workspace:/workspace/projects:Z \ kicad-mcp \ kicad
```

**In case of an MCP update, rebuild the container**

```shell
podman build --no-cache -t kicad-mcp .
```

## Metrics

The metrics kicad_*.py files need to be placed inside *$HOME/kicad_workspace* (there also live the KiCAD projects), so that they can be called with the KiCAD python version.

In the metrics.py file, those functions are called via subprocesses and can be used more freely.

## Harnesses

There a four harnesses, which have been used for evaluating the influence of context and roles on the performance of an LLM on PCB layout design:

- **Base Harness**: Single-Agent only basic text
- **Context Harness**: Single-Agent augmented text
- **Vision Harness**: Single-Agent text and vision capabilities
- **Multi Harness**: Multi-Agent harness with text and vision

The connection to the LLM is established via the OpenAI interface. Credentials should be stored in a private file, e.g. creds.py.

For a comparision, please consult the paper.