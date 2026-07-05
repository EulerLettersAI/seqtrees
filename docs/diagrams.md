# Diagrams

SeqTree keeps PlantUML source files in `docs/diagrams/` so architecture and
workflow diagrams can be rendered by documentation tooling or external PlantUML
services.

## PlantUML Sources

- [Sequential synthesis](diagrams/sequential_synthesis.puml)
- [Class overview](diagrams/class_overview.puml)

## Sequential Synthesis Preview

```plantuml
@startuml
start
:Fit marginal for first variable;
:Fit conditional trees for later variables;
:Sample first variable;
while (More variables?) is (yes)
  :Route partial row through next tree;
  :Sample next variable from leaf distribution;
endwhile (no)
:Return synthetic table;
stop
@enduml
```
