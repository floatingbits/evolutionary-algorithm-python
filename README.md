# Evolutionary Algorithm

Eine Python-Implementierung von evolutionären Algorithmen für komplexe Optimierungsprobleme.

Übersetzt aus dem [PHP-Original](https://github.com/floatingbits/evolutionary-algorithm) mit pythonischen Design-Patterns und moderner Type-Unterstützung.

## Features

- **Flexible Architektur**: Generics und Protocols für maximale Erweiterbarkeit
- **Pythonisch**: Dataclasses, Protocols, Type Hints (Python 3.10+)
- **Einfach**: Keine überflüssigen Abstraktionen - direkt nutzbar
- **Vollständig typisiert**: MyPy-kompatibel mit strict mode
- **Modular**: Selection, Mutation, Recombination, Evaluation frei kombinierbar

## Installation

```bash
# Entwicklungsmodus
pip install -e .

# Mit Dev-Abhängigkeiten (Tests, Linting)
pip install -e ".[dev]"
```

## Schnellstart

```python
from evolutionary_algorithm.evolution import Tournament
from evolutionary_algorithm.examples.job_assignment.solver import solve_example_problem

# Setup
evolver, specimen_gen, jobs, num_machines = solve_example_problem()
population = specimen_gen(50)

# Evolution ausführen
tournament = Tournament(
    evolver=evolver,
    population=population,
    rounds=50,
    cleanup_interval=49
)

result = tournament.run()
best = tournament.best_specimen
print(f"Best fitness: {best.fitness}")
```

## Beispiel ausführen

```bash
python examples/job_assignment.py
```

Das Job-Assignment-Beispiel demonstriert die Optimierung der Maschinenzuordnung:
- 33 Jobs auf 5 Maschinen verteilen
- Ziel: Minimierung der maximalen Fertigstellungszeit (Makespan)
- Verwendet Selection, Mutation und Crossover

## Architektur

### Core-Komponenten

- **Genotype**: Genetische Repräsentation (z.B. `SymbolArrayGenotype[int]`)
- **Phenotype**: "Reale" Interpretation des Genotyps
- **Specimen**: Individuum mit Genotype, Phenotype und Fitness
- **SpecimenCollection**: Population von Specimens

### Evolution-Pipeline

1. **Evaluation**: Phenotype-Generierung und Fitness-Berechnung
2. **Selection**: Auswahl der besten Individuals
3. **Replenishment**: Auffüllen durch Mutation und Recombination
4. **Cleanup**: Entfernen von Duplikaten (optional)

### Eigenes Problem implementieren

```python
from dataclasses import dataclass
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.selection import SimpleSelector
from evolutionary_algorithm.mutation import SimpleSymbolArrayMutator, CollectionMutator
from evolutionary_algorithm.recombination import SymbolArrayCrossoverRecombinator, CollectionRecombinator
from evolutionary_algorithm.randomizer import random_int

# 1. Definiere Phenotype
@dataclass(frozen=True)
class MyPhenotype:
    value: float

# 2. Phenotype Generator
def generate_phenotype(genotype: SymbolArrayGenotype[int]) -> MyPhenotype:
    return MyPhenotype(sum(genotype.symbols))

# 3. Evaluator
def evaluate(phenotype: MyPhenotype) -> Fitness:
    return Fitness(phenotype.value)

# 4. Evolver konfigurieren
evolver = Evolver(
    phenotype_generator=generate_phenotype,
    evaluator=evaluate,
    selector=SimpleSelector(survival_rate=0.3),
    mutators=[
        CollectionMutator(
            SimpleSymbolArrayMutator(
                mutation_rate=0.1,
                symbol_generator=lambda: random_int(0, 10)
            )
        )
    ],
    recombinators=[
        CollectionRecombinator(
            SymbolArrayCrossoverRecombinator(crossover_points=2)
        )
    ]
)
```

## Design-Philosophie

### Unterschiede zum PHP-Original

✅ **Verwendet:**
- `Protocol` statt Interfaces für Duck Typing
- `dataclass` für saubere Datenstrukturen
- `Callable` für Strategy Pattern
- Type Variables (`TypeVar`, `Generic`) für Typsicherheit
- Direkte Instantiierung statt Factory-Pattern (wo nicht nötig)

❌ **Entfernt:**
- Überflüssige Interface-Hierarchien
- Factory-Klassen ohne echten Mehrwert
- Getter/Setter (ersetzt durch `@property` wo nötig)
- Container-Traits (ersetzt durch Python's native Collections)

### Python-Idiome

- **Protocols statt Interfaces**: Flexibler, kein Zwang zur Vererbung
- **Functions als First-Class**: Evaluators, Generators können Funktionen sein
- **Dataclasses**: Immutable Phenotypes und Fitness mit automatischen Operatoren
- **Collection Protocol**: `__iter__`, `__len__`, `__getitem__` für native Integration

## Testing

```bash
# Tests ausführen
pytest

# Mit Coverage
pytest --cov=evolutionary_algorithm

# Type Checking
mypy src/
```

## Struktur

```
src/evolutionary_algorithm/
├── genotype/          # Genetische Repräsentationen
├── phenotype/         # Observable Charakteristiken
├── specimen/          # Individuen und Populationen
├── evaluation/        # Fitness-Bewertung
├── evolution/         # Evolver und Tournament
├── selection/         # Selektionsstrategien
├── mutation/          # Mutationsoperatoren
├── recombination/     # Crossover-Operatoren
├── randomizer/        # Zufallsgeneratoren (vereinfacht)
└── examples/          # Beispiel-Implementierungen
    └── job_assignment/

examples/
└── job_assignment.py  # Ausführbares Beispiel
```

## Lizenz

Übersetzt von Soeren Parton's PHP-Implementation.

## Nächste Schritte

Die Graph-Module (für Genetic Programming mit Baumstrukturen) sind noch nicht übersetzt.
Sie können bei Bedarf ergänzt werden.

Für die meisten Probleme reichen die SymbolArray-basierten Genotypes aus.
