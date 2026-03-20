# Burmese Hybrid ELIZA Chat UI

This project includes a local browser-based chat interface for a Burmese Hybrid ELIZA chatbot. The UI serves a simple web app from Python and connects to a rule-based ELIZA engine with optional LSTM-based emotion detection.

If the model checkpoint is available, the chatbot can load the Burmese emotion model. If not, it still works in rule-based mode.

## Main File

- `eliza/burmese_chat_ui.py` - local web server and chat UI
- `eliza/hybrid-eliza-improve-ver1.py` - chatbot logic and model loading
- `eliza/eliza_eq_mm_thiri_improve_v1.pth` - Burmese model checkpoint available in this repo (not in the git)

## Requirements

- Python 3.11 or newer
- Project dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

Run from the project root:

```bash
python eliza/burmese_chat_ui.py --model_path eliza/eliza_eq_mm_improve_v1.pth
```

Then open this in your browser:

```text
http://127.0.0.1:8765
```

## Run Without Model

If you want to use the chatbot without loading the LSTM checkpoint:

```bash
python eliza/burmese_chat_ui.py
```

In that case, the UI still works, but it falls back to rule-based replies only.

## Optional Arguments

```bash
python eliza/burmese_chat_ui.py --host 127.0.0.1 --port 8765 --lang mm --model_path eliza/eliza_eq_mm_improve_v1.pth
```

Available options:

- `--host` - server host, default is `127.0.0.1`
- `--port` - server port, default is `8765`
- `--lang` - `mm` or `en`
- `--model_path` - path to the `.pth` checkpoint

## Example

```bash
python eliza/burmese_chat_ui.py --host 0.0.0.0 --port 9000 --lang mm --model_path eliza/eliza_eq_mm_improve_v1.pth
```

Then open:

```text
http://127.0.0.1:9000
```


## Notes

- The recommended command includes `--model_path` because the default Burmese model filename inside the script does not match the checkpoint filename currently stored in this repository.
- Press `Enter` to send a message in the UI.
- Press `Shift + Enter` to add a new line.
