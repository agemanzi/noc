# Thermal Energy Game

A physics-based thermal energy management game with optimization capabilities. Players manage thermal systems while learning about heat transfer, energy efficiency, and optimal control strategies.

## 🎮 Game Overview

The Thermal Energy Game combines realistic thermal physics simulation with interactive gameplay and mathematical optimization. Players control thermal systems, manage energy flows, and optimize performance while competing against AI-driven "ghost" players that use advanced optimization algorithms.

## 🚀 Features

- **Physics-Based Simulation**: Realistic thermal dynamics and heat transfer modeling
- **Interactive GUI**: Intuitive sandbox environment for experimentation
- **Optimization Engine**: MILP solver for finding optimal strategies
- **AI Ghost Players**: Compete against mathematically optimal solutions
- **Data Visualization**: Real-time charts and performance analytics
- **Scenario Support**: Multiple thermal management challenges

## 📁 Project Structure

```
thermal_energy_game/
├── README.md
├── requirements.txt
├── src/
│   └── thermal_game/
│       ├── __init__.py
│       ├── gui/                 # User interface components
│       │   ├── app.py          # Main application window
│       │   ├── sandbox.py      # Interactive sandbox environment
│       │   ├── charts.py       # Data visualization
│       │   └── sprites.py      # Game graphics and animations
│       ├── engine/             # Game engine and physics
│       │   ├── thermal_physics.py  # Heat transfer simulation
│       │   ├── data_loader.py      # Scenario and data management
│       │   └── game_state.py       # Game state management
│       └── solver/             # Optimization components
│           ├── milp_solver.py      # Mathematical optimization
│           └── ghost_runner.py     # AI opponent logic
├── data/
│   └── scenarios/              # Game scenarios and test data
│       └── day01_prices_weather.csv
├── config/
│   └── game_config.yaml       # Game configuration settings
├── scripts/
│   ├── run_gui.py             # Launch the game
│   └── solve_optimal.py       # Run optimization analysis
├── models/                     # Saved models and checkpoints
└── outputs/                    # Generated results and reports
```

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd thermal_energy_game
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv thermal_env
   
   # On Windows:
   thermal_env\Scripts\activate
   
   # On macOS/Linux:
   source thermal_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python scripts/run_gui.py
   ```

## 🎯 Quick Start

### Launch the Game
```bash
python scripts/run_gui.py
```

### Run Optimization Analysis
```bash
python scripts/solve_optimal.py
```

### Customize Game Settings
Edit `config/game_config.yaml` to modify game parameters, physics constants, and scenario settings.

## 🧮 Core Components

### Thermal Physics Engine
- Heat conduction, convection, and radiation modeling
- Multi-material thermal properties
- Dynamic temperature distributions
- Energy balance calculations

### Optimization Solver
- Mixed-Integer Linear Programming (MILP)
- Real-time optimal control strategies
- Multi-objective optimization support
- Constraint handling for physical limitations

### Interactive GUI
- Drag-and-drop thermal system design
- Real-time temperature visualization
- Performance metrics and scoring
- Scenario selection and custom challenges

## 📊 Game Modes

1. **Sandbox Mode**: Free-form thermal system experimentation
2. **Challenge Mode**: Pre-designed scenarios with specific objectives
3. **Competition Mode**: Race against AI optimization algorithms
4. **Learning Mode**: Guided tutorials and physics demonstrations

## 🔧 Configuration

The game behavior can be customized through `config/game_config.yaml`:

- Physics simulation parameters
- GUI appearance and layout
- Optimization solver settings
- Scenario difficulty levels
- Performance metrics and scoring

## 📈 Data and Scenarios

Sample scenarios are provided in `data/scenarios/`. Each scenario includes:
- Initial thermal system configuration
- Environmental conditions (weather, prices)
- Optimization objectives and constraints
- Performance benchmarks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- Physics simulation based on established heat transfer principles
- Optimization algorithms implemented using PuLP
- GUI built with tkinter for cross-platform compatibility

## 📞 Support

For questions, issues, or feature requests, please open an issue on the project repository.

---

**Happy thermal gaming! 🌡️⚡**