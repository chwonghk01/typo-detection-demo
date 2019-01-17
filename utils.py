import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import seaborn as sns
import requests
import json
from bs4 import BeautifulSoup


def visualize_article_typo(article_id):
  visualize_typo(get_content(article_id, verbose=False)[1])


def visualize_typo(text):
  text, pred = check_typo(text)
  visualize_article(text, pred)


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


def visualize_article(text:str, pred, width=64):
    font_files = font_manager.findSystemFonts(fontpaths='./')
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


def _clean_me(html):
    soup = BeautifulSoup(html, 'lxml')
    for s in soup(['script', 'style']):
        s.decompose()
    return ' '.join(soup.stripped_strings)


def _parse_main_content(article, verbose=True):
    article = json.loads(article)

    content_type = article['contentType']
    teaser = ' '.join(article['teaser'])
    html_string = [_['htmlString'] for _ in article['blocks'] if 'htmlString' in _]
    main_content = ' '.join(_ for _ in html_string if _)
    main_content = teaser + ' ' + main_content

    # Get image captions for photostories
    if content_type == 'photostory':
        image_caption = [_['image']['caption'] for _ in article['blocks'] if 'image' in _ and _['image']]
        captions = ' '.join(_ for _ in image_caption if _)

        gallery = [_['images'] for _ in article['blocks'] if 'images' in _ and _['images'] and _['blockType'] == 'gallery']
        gallery_captions = ' '.join(itertools.chain(*[[_['caption'] for _ in _] for _ in gallery]))
        main_content += ' '.join([captions, gallery_captions])

    # Remove HTML tags
    return _clean_me(main_content)


def get_article(article_id, verbose=True, returnAsJson=False):
    if type(article_id) is 'str':
        article_id = int(float(article_id))
    response = requests.get('https://int-data.api.hk01.com/v2/articles/{0:d}'.format(article_id))
    if response.status_code != 200:
        if verbose:
            print('Error for article id = {0:d}, status code = {1:d}'.format(article_id, response.status_code))
        return None, response.status_code

    if returnAsJson:
        return response.json(), response.status_code
    return response.content.decode('utf-8'), response.status_code


def get_content(article_id, verbose=True):
    """
    Get article's main content from latest version of API by article_id
    Return a tuple of (article_id, main_content, status_code). For non-200 status code,
    main_content will be a space.
    """
    article, status_code = get_article(article_id, verbose=verbose)
    if status_code != 200:
        return article_id, '', status_code

    main_content = _parse_main_content(article)
    return article_id, main_content, status_code
