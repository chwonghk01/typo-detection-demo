import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import seaborn as sns
import requests
import json

def visualize_typo(font_path, text):
  text, pred = check_typo(text)
  visualize_article(font_path, text, pred)


def visualize(asset_path, inp, truth, pred=None, display_rows=1):
  font_files = font_manager.findSystemFonts(fontpaths=asset_path)
  font_list = font_manager.createFontList(font_files)
  font_manager.fontManager.ttflist.extend(font_list)
  sns.set_style("darkgrid",{"font.sans-serif":['SimHei', 'Arial']})

  assert inp.shape[0] == truth.shape[0]
  seq_len = inp.shape[1]

  fig, ax = plt.subplots(figsize=(24 * seq_len // 64, display_rows / 3))
  sns.heatmap(truth[:display_rows], \
    annot=np.array([[c for c in row] for row in inp[:display_rows]]), \
    fmt='', cmap="YlGnBu", ax=ax, vmin=0, vmax=1)

  # plt.savefig('visualize.png')
  plt.show()

  if pred is not None:
    fig, ax = plt.subplots(figsize=(24 * seq_len // 64,display_rows / 3))
    sns.heatmap(pred[:display_rows], \
      annot=np.array([[c for c in row] for row in inp[:display_rows]]), \
      fmt='', cmap="YlGnBu", ax=ax, vmin=0, vmax=1)

    # plt.savefig('visualize_2.png')
    plt.show()


def _pad_to_fixed_width(list, width):
    """
    >>> _pad_to_fixed_width()
    """
    l = np.array(list)
    pad_count = (width - len(l) % width)
    pad_count = pad_count if pad_count != width else 0
    p = np.pad(l, (0, pad_count), 'constant')
    return np.reshape(p, (len(p) // width, width))


def visualize_article(font_path, text:str, pred, width=64):
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)
    sns.set_style("darkgrid",{"font.sans-serif":['SimHei', 'Arial']})

    assert len(text) == len(pred)
    text = [ch for ch in text]
    text = _pad_to_fixed_width(text, width)
    pred = _pad_to_fixed_width(pred, width)

    assert text.shape[0] == pred.shape[0]
    assert text.shape[1] == pred.shape[1]

    fig, ax = plt.subplots(figsize=(24 * text.shape[1] // 64, text.shape[0] / 3))
    sns.heatmap(pred, annot=text, fmt='', cmap="YlGnBu", ax=ax, vmin=0, vmax=1)


def check_typo(text):
  resp = requests.post('https://us-east1-data-poc-227904.cloudfunctions.net/typo-detection', json=[{"text": text}])
  result = json.loads(resp.content)
  return result[0]['text'], result[0]['predictions']
