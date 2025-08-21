# UML Diagrams

To generate diagrams for this book, I've used [D2](https://d2lang.com/). D2
stands for Declarative Diagramming and is a declarative scripting language for
generating diagrams, such as UML class and sequence diagrams.

To generate the UML diagrams of this book, you'll need to have D2 installed on
your machine. Use the command found below to perform the installation:

```sh
curl -fsSL https://d2lang.com/install.sh | sh -s --
```

## Generating Diagrams

After installing `d2` on your machine, you can now generate any UML image from
the `*.d2` files contained in this folder.

```sh
# Generate SVG (vector format, good for web)
d2 uml/ch04/llm_agent.d2 uml/ch04/llm_agent.svg --sketch

# Generate PNG (raster format, good for print)
d2 uml/ch04/llm_agent.d2 uml/ch04/llm_agent.png --sketch

# Watch mode - automatically regenerates when .d2 file changes
d2 -w uml/ch04/llm_agent.d2 uml/ch04/llm_agent.png --sketch

# High resolution for print (2x scale)
d2 uml/ch04/llm_agent.d2 uml/ch04/llm_agent.png --sketch --scale 2
```
