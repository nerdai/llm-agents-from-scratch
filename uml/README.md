# UML Diagrams

To generate diagrams for this book, I've used [PlantUML](https://plantuml.com/). PlantUML is a mature, open-source tool for creating UML diagrams from plain text descriptions, supporting class diagrams, sequence diagrams, activity diagrams, and more.

## Installation

### Prerequisites

PlantUML requires Java to be installed on your machine.

### Installing PlantUML

#### Option 1: Package Manager (may have older version)**

```sh
# Ubuntu/Debian
sudo apt install plantuml

# macOS
brew install plantuml
```

#### Option 2: Latest JAR (recommended)**

```sh
# Download the latest PlantUML JAR
wget https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar -O ~/plantuml.jar

# Add alias to your shell configuration (~/.bashrc or ~/.zshrc)
echo "alias plantuml-new='java -jar ~/plantuml.jar'" >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

## Generating Diagrams

After installing PlantUML, you can generate UML images from the `*.puml` files in this folder.

### Individual Files

```sh
# Generate PNG (default, good for print)
plantuml-new uml/ch04/llm_agent_class.puml

# Generate SVG (vector format, good for web)
plantuml-new -tsvg uml/ch04/llm_agent_class.puml

# High resolution for print
plantuml-new -SDPI=300 uml/ch04/llm_agent_class.puml
```

### Batch Generation

```sh
# Generate all diagrams using make
make diagrams      # PNG files
make diagrams-svg  # SVG files

# Or manually generate all files
plantuml-new -SDPI=300 uml/**/*.puml
```

## Directory Structure

```sh
uml/
├── common/
│   └── book-clean.puml     # Shared styling
├── ch04/
│   ├── llm_agent_class.puml
│   └── llm_agent_sequence.puml
└── rendered/               # Generated images
    └── ch04/
        ├── llm_agent_class.png
        ├── llm_agent_class.svg
        ├── llm_agent_sequence.png
        └── llm_agent_sequence.svg
```
