# TherMOS GUI — User Guide

The TherMOS GUI is a local web application (Streamlit). All parameters are edited in the browser and stored as JSON;
the input data are still read from the workbooks. The GUI runs on your own machine only (`http://localhost:8501`).

## 1. Installation and start

```
pip install -r gui/requirements.txt
python start_gui.py                      # opens http://localhost:8501
```

Stop the GUI with `Ctrl+C` in the terminal. Server settings (port, localhost only, theme) are in
`.streamlit/config.toml`.

## 2. Layout

| Area | Content |
|---|---|
| **Sidebar** | Expert mode switch, configuration manager (load / save / download / import / reset), runs of this session, navigation |
| **Setup** | Run settings, input files, shared parameters |
| **Input data** | Every sheet of the input workbooks |
| **Operating modes** | One page per mode with the views *Workflow*, *Parameters*, *Inputs*, *Run*, *Results* |
| **Results browser** | All results folders of all projects, incl. runs started with the start scripts |
| **Help** | This guide and the parameter reference |

## 3. Configurations

The configuration is the nested dictionary `setup` of the toolkit. *Saved configs → Load* reads `config/<name>.json`,
*Save* writes it, *Import* reads a `.json` or a config workbook `.xlsx`. For a run the GUI writes the configuration
into `config/_gui_run.xlsx` (a copy of the template workbook) and the toolkit loads it as usual.

If parameters are added to the config workbook, regenerate the schema and defaults with `python gui/build_schema.py`.

## 4. Running a simulation

The *Run* view shows a summary and the button **Run**. The console output is streamed; figures are captured and shown
under *Results › Figures & files*. Runs are stored as `gui_run.pkl` / `gui_run.json` in their results folder.

## 5. Troubleshooting

| Problem | Remedy |
|---|---|
| Port 8501 in use | Stop the other instance or change `server.port` in `.streamlit/config.toml` |
| Changes in the code are not visible | Restart `start_gui.py` |
| A config from an older version misses parameters | Missing parameters are filled with the defaults on loading |
