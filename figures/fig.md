
# CONVERT


```
import yaml
import re
import os
from functools import wraps

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def is_callable(arg):
    return hasattr(arg, "__call__")

def parse_file(file, register, run_parallel=True):

    assert os.path.exists(file), f"{file}: File not found"

    opt = None
    settings_loc = 0

    r_settings_order = []
    shift = 0

    def init_settings():
        nonlocal opt, settings_loc, shift
        opt = Struct()
        opt.out = "Readme"
        settings_loc = 0
        shift = 0


    def find_settings_order(match):
        r_settings_order.append((match.start(), match.group()))
        return ''

    def apply_settings(stream):
        out_file = opt.out
        for k, v in yaml.load(stream)['nbconv'].items():
            setattr(opt, k, v)
        opt.out = opt.out.split('.')[0]
        # if opt.out != out_file:
        #     return f"XXX FILESTART:{out_file}"
        # return ''
    
    def re_wrapper(f):
        @wraps(f)
        def _func(match):
            nonlocal settings_loc, shift
            len_before = len(match.group())
            settings = r_settings_order[settings_loc] if settings_loc < len(r_settings_order) else None
            if settings and settings[0] + shift < match.start():
                print(match.start(), "applying", settings)
                apply_settings(settings[1])
                settings_loc += 1
            out = f(match, opt)
            shift += len(out) - len_before
            # if settings_loc < len(r_settings_order):
            #     settings = r_settings_order[settings_loc]
                # r_settings_order[settings_loc] = (settings[0] + shift, settings[1])
        return _func

    r_start = re.compile(r"(?<=nbconv: START$)[\s\S]+?(?=\nnbconv: END|\Z)", re.MULTILINE)
    r_settings = re.compile(r"^nbconv:\s*$\n([^\S\r\n]*\w+:\s*\S+$\n)+", re.MULTILINE)

    # raw_register = [(r"^nbconv:\s*$\n([^\S\r\n]*\w+:\s*\S+$\n)+", parse_settings)] 
    # raw_register = [(r, re_wrapper(dummy)) for r, f in register if is_callable(f)]
    raw_register = [(r, re_wrapper(f)) if is_callable(f) else (r, f) for r, f in register]
    r_register = [(re.compile(r, re.MULTILINE), f) for r, f in raw_register]
    # r_any = re.compile(f"({'|'.join(r for r, f in raw_register)})", re.MULTILINE)



    # def replace_any(md_match):
    #     return out
    

    with open(file, 'r') as f:
        full_text = f.read()
        start = re.search(r_start, full_text)

        if not start:
            print("Couldn't find start 'nbconv: START'")
            md = full_text
        else:
            md = start.group()
        
        for r, f in r_register:
            print("PARSING", f)
            init_settings()
            re.sub(r_settings, find_settings_order, md)
            print(r_settings_order)
            # print(md[r_settings_order[0][0]:r_settings_order[1][0] + 40])
            md = re.sub(r, f, md)
        
        md = re.sub(r_settings, '', md)
        
        if md.strip():
            print(f"writing to {opt.out}")
            with open(opt.out + ".md", 'w') as f:
                f.write(md)



%cd /content/Thesis
cwd = os.getcwd()
url_base = "https://raw.githubusercontent.com/willisk/Thesis/master"
# cwd = "/content/Thesis"
md_file = 'fig.md'

def sub_images(match, opt):
    file_name_in = "".join(match.group(i) for i in [1, 2, 3])
    prefix = match.group(1)
    md_out_base = opt.out.split('.')[0]
    file_name_out_i = f"{prefix}{md_out_base}_{{}}.png"
    i = 0
    while os.path.exists(file_name_out_i.format(i)):
        i += 1
    file_name_out = file_name_out_i.format(i)
    # !mv {file_name_in} {file_name_out}
    link_out = file_name_out.replace(cwd, url_base)
    print(f"{match.start()} file: {opt.out} im: {opt.images} IN: {file_name_in} OUT: {file_name_out} URL: {link_out}")
    return f"![png]({link_out})"

register = [
        (r"(?:!\[png\]\()(.*?)([^/\n]+)(\.png)(?:\))", sub_images),
        # ("    #", "#"),
]

# !jupyter nbconvert --to markdown \
#     '/content/drive/My Drive/Colab Notebooks/Project OK.ipynb' \
#     --output {cwd}/figures/fig.md

parse_file(f"{cwd}/figures/fig.md", register)
```

    /content/Thesis
    PARSING <function sub_images at 0x7fe730f66730>
    [(2, "nbconv:\n    out: 'Readme.md'\n    images: 'reconstruction_cifar10'\n"), (967, "nbconv:\n    images: 'reconstruction_mnist'\n"), (1734, "nbconv:\n    images: 'inversion_cifar10'\n"), (17250, "nbconv:\n    images: 'inversion_mnist'\n"), (17726, "nbconv:\n    out: 'Tests'\n")]
    4390 applying (2, "nbconv:\n    out: 'Readme.md'\n    images: 'reconstruction_cifar10'\n")
    4390 file: Readme im: reconstruction_cifar10 IN: /content/Thesis/figures/fig_24_6.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    4579 applying (967, "nbconv:\n    images: 'reconstruction_mnist'\n")
    4579 file: Readme im: reconstruction_mnist IN: /content/Thesis/figures/fig_24_10.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    7326 applying (1734, "nbconv:\n    images: 'inversion_cifar10'\n")
    7326 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_3.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    7500 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_7.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    7674 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_11.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    7849 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_15.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8024 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_19.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8199 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_23.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8366 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_28.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8415 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_29.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8480 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_31.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8743 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_35.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    8918 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_39.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9093 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_43.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9268 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_47.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9443 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_51.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9618 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_55.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9782 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_60.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9831 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_61.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9880 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_62.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    9945 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_64.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    10213 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_68.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    10390 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_72.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    10564 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_76.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    10741 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_80.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    10914 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_84.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11089 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_88.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11257 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_93.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11306 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_94.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11355 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_95.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11420 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_97.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11689 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_101.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    11867 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_105.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12045 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_109.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12223 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_113.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12401 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_117.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12579 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_121.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12745 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_126.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12795 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_127.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12845 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_128.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12895 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_129.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    12961 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_131.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    13231 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_135.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    13406 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_139.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    13581 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_143.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    13757 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_147.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    13933 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_151.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14109 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_155.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14277 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_160.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14327 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_161.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14377 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_162.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14427 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_163.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14493 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_165.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14754 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_169.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    14927 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_173.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15100 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_177.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15273 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_181.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15446 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_185.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15619 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_189.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15784 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_194.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15834 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_195.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15884 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_196.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    15950 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_198.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    16217 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_202.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    16393 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_206.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    16569 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_210.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    16745 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_214.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    16921 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_218.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    17097 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_26_222.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    19353 file: Readme im: inversion_cifar10 IN: /content/Thesis/figures/fig_32_4.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    21548 applying (17250, "nbconv:\n    images: 'inversion_mnist'\n")
    21548 file: Readme im: inversion_mnist IN: /content/Thesis/figures/fig_32_28.png OUT: /content/Thesis/figures/Readme_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Readme_0.png
    21904 applying (17726, "nbconv:\n    out: 'Tests'\n")
    21904 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_32.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    22261 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_36.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    22621 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_40.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    22974 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_44.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    23330 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_48.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    23688 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_52.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    24049 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_56.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    24408 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_32_60.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    26656 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_37_1.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    26857 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_37_5.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    26933 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_37_7.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    27567 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_41_2.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    27615 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_41_3.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    27816 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_41_7.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    27989 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_41_9.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    28444 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_2.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    28492 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_3.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    28776 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_7.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    28934 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_9.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    28982 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_10.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    29031 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_42_11.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    29440 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_44_2.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    29724 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_44_6.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    29896 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_44_8.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    30325 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_46_2.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    30373 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_46_3.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    30657 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_46_7.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    30829 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_46_9.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    31325 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_48_2.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    31609 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_48_6.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png
    31725 file: Tests im: inversion_mnist IN: /content/Thesis/figures/fig_48_8.png OUT: /content/Thesis/figures/Tests_0.png URL: https://raw.githubusercontent.com/willisk/Thesis/master/figures/Tests_0.png



```
rgx_img_rename = re.compile(r"([^/]+)(?=\.png)")
re.sub(rgx_img_rename, "blah", "![png](/content/Thesis/figures/README/dslfjlskjf.png")
```




    '![png](/content/Thesis/figures/README/blah.png'




```
import re
in_run = False
in_md = False
with open("readme.tmp") as f, open("README.md", "w") as readme, open("TESTS.md", "w") as tests:
    out = readme
    for line in f:
        if re.search("^# MAIN", line):
            in_md = True
            continue
        elif re.search("^# TESTS", line):
            out = tests
            continue
        elif in_md:
            line = line.replace("![png](/content/Thesis/figures/README/",
                         "![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/")
            if line[:5] == "    #":
                line = line[4:]
            if line[:8] == "    HBox":
                continue
            line = re.sub("^\%run", "python", line)
            out.write(line)

%cd /content/Thesis/
!rm readme.tmp
```


```
import os
cwd = os.getcwd()

%cd {cwd}
# !rm -r figures/README
!jupyter nbconvert --to markdown \
    '/content/drive/My Drive/Colab Notebooks/Project OK.ipynb' \
    --output {cwd}/figures/fig.md
# !mv {cwd}/figures/README/README.md {cwd}/readme.tmp

```

    /content/Thesis
    [NbConvertApp] Converting notebook /content/drive/My Drive/Colab Notebooks/Project OK.ipynb to markdown
    [NbConvertApp] Support files will be in /content/Thesis/figures/fig_files/
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Making directory /content/Thesis/figures
    [NbConvertApp] Writing 42916 bytes to /content/Thesis/figures/fig.md


# PUSH


```
%cd /content/Thesis/
!git config --global user.email "willis@campus.tu-berlin.de"
!git config --global user.name "Kurt Willis"
!git remote rm origin
!git remote add origin https://willisk:%3F%21@github.com/willisk/Thesis.git
!git push --set-upstream origin master
!git add .
!git commit -m "update readme"
!git status
!git push -u origin
```

    [Errno 2] No such file or directory: '/content/Thesis/'
    /content
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git
    fatal: not a git repository (or any of the parent directories): .git


# PULL


```
!git clone https://github.com/willisk/Thesis /content/Thesis
%cd /content/Thesis
!git pull
!git reset --hard origin/master
```

    Cloning into '/content/Thesis'...
    remote: Enumerating objects: 101, done.[K
    remote: Counting objects: 100% (101/101), done.[K
    remote: Compressing objects: 100% (65/65), done.[K
    remote: Total 3180 (delta 61), reused 72 (delta 36), pack-reused 3079[K
    Receiving objects: 100% (3180/3180), 2.63 MiB | 23.21 MiB/s, done.
    Resolving deltas: 100% (2096/2096), done.
    /content/Thesis
    Already up to date.
    HEAD is now at efb7fd3 inversion upd



```
from google.colab import drive
drive.mount('/content/drive')
```

nbconv: START

nbconv:
    out: 'Readme.md'
    images: 'reconstruction_cifar10'

# ->Start
# RECONSTRUCTION

## CIFAR10


```
# r_distort=0.2

!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=1024 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=100 \
-r_distort_level=0.2 \
-r_block_width=16 \
-r_block_depth=4 \
--plot_ideal \
-show_after=20 \
-seed=1 \

# -size_A=-1 \
# -size_B=512 \
# -batch_size=256 \
# -seed=23456

# --reset_stats \
```


```
# pretty samples
# r_distort=0.1

!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=1024 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=100 \
-r_distort_level=0.1 \
--plot_ideal \
-show_after=20 \
-seed=1
```

    Already up to date.
    HEAD is now at 8b1cf4e depth
    # Testing reconstruction methods
    # on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=1
    nn_lr=0.01
    nn_steps=100
    batch_size=128
    n_random_projections=512
    inv_lr=0.1
    inv_steps=100
    f_reg=0.0
    f_crit=1.0
    f_stats=100.0
    size_A=1024
    size_B=512
    show_after=20
    r_distort_level=0.1
    r_block_depth=4
    r_block_width=4
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=True
    scale_each=False 
    
    Running on 'cuda'
    
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /content/Thesis/data/cifar-10-python.tar.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting /content/Thesis/data/cifar-10-python.tar.gz to /content/Thesis/data
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.6%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 78.6%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-512.pt.
    
    ground truth:



![png](/content/Thesis/figures/fig_15_3.png)


    
    
    distorted:



![png](/content/Thesis/figures/fig_15_5.png)


    
    
    
    
    ## Method: CRITERION
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_7.png)


    


     20%|██        |20.0/100 [01:04<04:16, 3.20s/epoch, accuracy=0.852, ideal=0.427, loss=0.642, psnr=19.2, |grad|=1.58]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_11.png)


    


     40%|████      |40.0/100 [02:08<03:12, 3.21s/epoch, accuracy=0.93, ideal=0.364, loss=0.437, psnr=18.9, |grad|=1.08]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_15.png)


    


     60%|██████    |60.0/100 [03:12<02:08, 3.20s/epoch, accuracy=0.922, ideal=0.428, loss=0.522, psnr=20.6, |grad|=1.29]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_19.png)


    


     80%|████████  |80.0/100 [04:16<01:04, 3.20s/epoch, accuracy=0.961, ideal=0.369, loss=0.396, psnr=20.4, |grad|=0.7]

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_23.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.20s/epoch, accuracy=0.992, ideal=0.342, loss=0.353, psnr=21.3, |grad|=1.21]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_27.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.21s/epoch, accuracy=0.992, ideal=0.342, loss=0.353, psnr=21.3, |grad|=1.21]

    


    



![png](/content/Thesis/figures/fig_15_32.png)



![png](/content/Thesis/figures/fig_15_33.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_35.png)


    
    Results:
    	loss: 1.540
    	average PSNR: 18.419 | (distorted: 23.959)
    	rel. l2 reconstruction error: 22.265 | (distorted: 7.058)
    	nn accuracy: 95.1 %
    	nn validation set accuracy: 82.6 %
    	nn verifier accuracy: 70.3 %
    
    
    
    ## Method: NN
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_37.png)


    


     20%|██        |20.0/100 [01:03<04:15, 3.19s/epoch, accuracy=0.828, ideal=1.34, loss=1.69, psnr=16.3, |grad|=2.99]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_41.png)


    


     40%|████      |40.0/100 [02:08<03:12, 3.20s/epoch, accuracy=0.844, ideal=1.37, loss=1.36, psnr=19.4, |grad|=3.33]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_45.png)


    


     60%|██████    |60.0/100 [03:12<02:08, 3.20s/epoch, accuracy=0.875, ideal=1.13, loss=1.26, psnr=18.3, |grad|=4.35]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_49.png)


    


     80%|████████  |80.0/100 [04:16<01:03, 3.19s/epoch, accuracy=0.883, ideal=1.46, loss=1.28, psnr=19.6, |grad|=2.33]  

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_53.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.19s/epoch, accuracy=0.852, ideal=1.34, loss=1.23, psnr=15.7, |grad|=1.85] 

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_57.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.21s/epoch, accuracy=0.852, ideal=1.34, loss=1.23, psnr=15.7, |grad|=1.85]

    


    



![png](/content/Thesis/figures/fig_15_62.png)



![png](/content/Thesis/figures/fig_15_63.png)



![png](/content/Thesis/figures/fig_15_64.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_66.png)


    
    Results:
    	loss: 4.628
    	average PSNR: 16.918 | (distorted: 23.884)
    	rel. l2 reconstruction error: 16.275 | (distorted: 7.058)
    	nn accuracy: 89.5 %
    	nn validation set accuracy: 76.4 %
    	nn verifier accuracy: 63.3 %
    
    
    
    ## Method: NN CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_68.png)


    


     20%|██        |20.0/100 [01:04<04:18, 3.24s/epoch, accuracy=0.898, ideal=1, loss=1.42, psnr=18.7, |grad|=3.29]    

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_72.png)


    


     40%|████      |40.0/100 [02:09<03:13, 3.23s/epoch, accuracy=0.898, ideal=0.978, loss=1.44, psnr=20.1, |grad|=4.65]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_76.png)


    


     60%|██████    |60.0/100 [03:14<02:09, 3.23s/epoch, accuracy=0.938, ideal=1.23, loss=1.4, psnr=19.7, |grad|=3.96]  

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_80.png)


    


     80%|████████  |80.0/100 [04:18<01:04, 3.23s/epoch, accuracy=0.93, ideal=1.18, loss=1.42, psnr=19, |grad|=4.92]   

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_84.png)


    


    100%|██████████|100.0/100 [05:23<00:00, 3.23s/epoch, accuracy=0.922, ideal=0.942, loss=1.36, psnr=21.9, |grad|=5.52]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_88.png)


    


    100%|██████████|100.0/100 [05:23<00:00, 3.24s/epoch, accuracy=0.922, ideal=0.942, loss=1.36, psnr=21.9, |grad|=5.52]

    


    



![png](/content/Thesis/figures/fig_15_93.png)



![png](/content/Thesis/figures/fig_15_94.png)



![png](/content/Thesis/figures/fig_15_95.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_97.png)


    
    Results:
    	loss: 4.580
    	average PSNR: 22.418 | (distorted: 23.941)
    	rel. l2 reconstruction error: 20.167 | (distorted: 7.058)
    	nn accuracy: 96.1 %
    	nn validation set accuracy: 84.6 %
    	nn verifier accuracy: 72.1 %
    
    
    
    ## Method: NN ALL
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_99.png)


    


     20%|██        |20.0/100 [01:07<04:29, 3.37s/epoch, accuracy=0.852, ideal=2.37, loss=1.5, psnr=21.8, |grad|=8.83]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_103.png)


    


     40%|████      |40.0/100 [02:14<03:21, 3.36s/epoch, accuracy=0.93, ideal=1.32, loss=1.14, psnr=22.4, |grad|=7.78] 

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_107.png)


    


     60%|██████    |60.0/100 [03:22<02:14, 3.36s/epoch, accuracy=0.953, ideal=1.54, loss=1.18, psnr=22.8, |grad|=8.8]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_111.png)


    


     80%|████████  |80.0/100 [04:29<01:07, 3.36s/epoch, accuracy=0.875, ideal=2.14, loss=1.2, psnr=23.2, |grad|=9.31] 

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_115.png)


    


    100%|██████████|100.0/100 [05:37<00:00, 3.36s/epoch, accuracy=0.945, ideal=1.26, loss=1.25, psnr=25.8, |grad|=10.6]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_119.png)


    


    100%|██████████|100.0/100 [05:37<00:00, 3.37s/epoch, accuracy=0.945, ideal=1.26, loss=1.25, psnr=25.8, |grad|=10.6]

    


    



![png](/content/Thesis/figures/fig_15_124.png)



![png](/content/Thesis/figures/fig_15_125.png)



![png](/content/Thesis/figures/fig_15_126.png)



![png](/content/Thesis/figures/fig_15_127.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_129.png)


    
    Results:
    	loss: 4.213
    	average PSNR: 24.563 | (distorted: 23.962)
    	rel. l2 reconstruction error: 10.973 | (distorted: 7.058)
    	nn accuracy: 96.1 %
    	nn validation set accuracy: 84.4 %
    	nn verifier accuracy: 73.0 %
    
    
    
    ## Method: NN ALL CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_131.png)


    


     20%|██        |20.0/100 [01:40<06:39, 4.99s/epoch, accuracy=0.938, ideal=3.02, loss=3.12, psnr=23.6, |grad|=3.05]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_135.png)


    


     40%|████      |40.0/100 [03:20<05:01, 5.02s/epoch, accuracy=0.93, ideal=3.13, loss=2.91, psnr=21.9, |grad|=4.93]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_139.png)


    


     60%|██████    |60.0/100 [05:00<03:19, 5.00s/epoch, accuracy=0.961, ideal=3.06, loss=3.01, psnr=24.5, |grad|=3.28]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_143.png)


    


     80%|████████  |80.0/100 [06:40<01:40, 5.01s/epoch, accuracy=0.977, ideal=3.06, loss=2.75, psnr=24.3, |grad|=3.02]

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_147.png)


    


    100%|██████████|100.0/100 [08:20<00:00, 5.03s/epoch, accuracy=0.953, ideal=3.22, loss=2.76, psnr=24, |grad|=3.96] 

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_151.png)


    


    100%|██████████|100.0/100 [08:20<00:00, 5.01s/epoch, accuracy=0.953, ideal=3.22, loss=2.76, psnr=24, |grad|=3.96]

    


    



![png](/content/Thesis/figures/fig_15_156.png)



![png](/content/Thesis/figures/fig_15_157.png)



![png](/content/Thesis/figures/fig_15_158.png)



![png](/content/Thesis/figures/fig_15_159.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_161.png)


    
    Results:
    	loss: 11.311
    	average PSNR: 23.547 | (distorted: 23.908)
    	rel. l2 reconstruction error: 18.713 | (distorted: 7.058)
    	nn accuracy: 95.1 %
    	nn validation set accuracy: 83.3 %
    	nn verifier accuracy: 69.1 %
    
    
    
    ## Method: RP
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_163.png)


    


     20%|██        |20.0/100 [01:03<04:15, 3.20s/epoch, accuracy=0.477, ideal=801, loss=645, psnr=16.7, |grad|=210]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_167.png)


    


     40%|████      |40.0/100 [02:07<03:11, 3.20s/epoch, accuracy=0.516, ideal=881, loss=734, psnr=18.2, |grad|=1.17e+3]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_171.png)


    


     60%|██████    |60.0/100 [03:11<02:07, 3.20s/epoch, accuracy=0.461, ideal=1.02e+3, loss=695, psnr=14.8, |grad|=340]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_175.png)


    


     80%|████████  |80.0/100 [04:15<01:03, 3.20s/epoch, accuracy=0.453, ideal=726, loss=661, psnr=14.2, |grad|=425]

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_179.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.19s/epoch, accuracy=0.453, ideal=1.08e+3, loss=635, psnr=18.7, |grad|=425]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_183.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.20s/epoch, accuracy=0.453, ideal=1.08e+3, loss=635, psnr=18.7, |grad|=425]

    


    



![png](/content/Thesis/figures/fig_15_188.png)



![png](/content/Thesis/figures/fig_15_189.png)



![png](/content/Thesis/figures/fig_15_190.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_192.png)


    
    Results:
    	loss: 2554.835
    	average PSNR: 17.473 | (distorted: 23.911)
    	rel. l2 reconstruction error: 49.880 | (distorted: 7.058)
    	nn accuracy: 44.5 %
    	nn validation set accuracy: 40.3 %
    	nn verifier accuracy: 30.9 %
    
    
    
    ## Method: RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_194.png)


    


     20%|██        |20.0/100 [01:05<04:21, 3.27s/epoch, accuracy=0.406, ideal=2.35e+3, loss=2.1e+3, psnr=18.3, |grad|=174] 

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_198.png)


    


     40%|████      |40.0/100 [02:10<03:15, 3.26s/epoch, accuracy=0.375, ideal=2.48e+3, loss=1.98e+3, psnr=14.6, |grad|=463]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_202.png)


    


     60%|██████    |60.0/100 [03:16<02:10, 3.27s/epoch, accuracy=0.367, ideal=2.42e+3, loss=2.04e+3, psnr=18.1, |grad|=417]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_206.png)


    


     80%|████████  |80.0/100 [04:21<01:05, 3.26s/epoch, accuracy=0.422, ideal=2.3e+3, loss=1.95e+3, psnr=16.4, |grad|=205]

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_210.png)


    


    100%|██████████|100.0/100 [05:27<00:00, 3.27s/epoch, accuracy=0.367, ideal=2.57e+3, loss=2.06e+3, psnr=16.9, |grad|=222]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_214.png)


    


    100%|██████████|100.0/100 [05:27<00:00, 3.27s/epoch, accuracy=0.367, ideal=2.57e+3, loss=2.06e+3, psnr=16.9, |grad|=222]

    


    



![png](/content/Thesis/figures/fig_15_219.png)



![png](/content/Thesis/figures/fig_15_220.png)



![png](/content/Thesis/figures/fig_15_221.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_223.png)


    
    Results:
    	loss: 8066.062
    	average PSNR: 16.492 | (distorted: 23.872)
    	rel. l2 reconstruction error: 31.708 | (distorted: 7.058)
    	nn accuracy: 36.1 %
    	nn validation set accuracy: 33.9 %
    	nn verifier accuracy: 27.1 %
    
    
    
    ## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_15_225.png)


    


     20%|██        |20.0/100 [01:41<06:48, 5.11s/epoch, accuracy=0.906, ideal=68.9, loss=57.5, psnr=18.7, |grad|=14.6]

    
    epoch 20:



![png](/content/Thesis/figures/fig_15_229.png)


    


     40%|████      |40.0/100 [03:23<05:05, 5.10s/epoch, accuracy=0.898, ideal=64.2, loss=58, psnr=22.4, |grad|=8.51]

    
    epoch 40:



![png](/content/Thesis/figures/fig_15_233.png)


    


     60%|██████    |60.0/100 [05:05<03:23, 5.08s/epoch, accuracy=0.898, ideal=63.9, loss=60.2, psnr=21.9, |grad|=15.1]

    
    epoch 60:



![png](/content/Thesis/figures/fig_15_237.png)


    


     80%|████████  |80.0/100 [06:46<01:41, 5.06s/epoch, accuracy=0.922, ideal=68.2, loss=56.8, psnr=20.2, |grad|=7.5] 

    
    epoch 80:



![png](/content/Thesis/figures/fig_15_241.png)


    


    100%|██████████|100.0/100 [08:28<00:00, 5.12s/epoch, accuracy=0.93, ideal=58.9, loss=53.7, psnr=22, |grad|=14.9]

    
    epoch 100:



![png](/content/Thesis/figures/fig_15_245.png)


    


    100%|██████████|100.0/100 [08:28<00:00, 5.09s/epoch, accuracy=0.93, ideal=58.9, loss=53.7, psnr=22, |grad|=14.9]

    


    



![png](/content/Thesis/figures/fig_15_250.png)



![png](/content/Thesis/figures/fig_15_251.png)



![png](/content/Thesis/figures/fig_15_252.png)



![png](/content/Thesis/figures/fig_15_253.png)


    Inverted:
    128 / 512 



![png](/content/Thesis/figures/fig_15_255.png)


    
    Results:
    	loss: 233.005
    	average PSNR: 21.296 | (distorted: 23.892)
    	rel. l2 reconstruction error: 8.410 | (distorted: 7.058)
    	nn accuracy: 91.2 %
    	nn validation set accuracy: 79.7 %
    	nn verifier accuracy: 65.0 %
    
    # Summary
    =========
    
    
    baseline       acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    -----------------------------------------------------------
    B (original)   0.96  0.89      0.80      --        --      
    B (distorted)  0.56  0.51      0.44      23.89     7.06    
    A              0.97  --        0.79      --        --      
    
    Reconstruction methods:
    
    method          acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    ------------------------------------------------------------
    CRITERION       0.95  0.83      0.70      18.42     22.26   
    NN              0.89  0.76      0.63      16.92     16.28   
    NN CC           0.96  0.85      0.72      22.42     20.17   
    NN ALL          0.96  0.84      0.73      24.56     10.97   
    NN ALL CC       0.95  0.83      0.69      23.55     18.71   
    RP              0.45  0.40      0.31      17.47     49.88   
    RP CC           0.36  0.34      0.27      16.49     31.71   
    NN ALL + RP CC  0.91  0.80      0.65      21.30     8.41    


## MNIST



nbconv:
    images: 'reconstruction_mnist'


```
!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=MNIST \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=-1 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=0.001 \
-r_distort_level=0.5 \
--plot_ideal \
-show_after=20 \
```

    Already up to date.
    HEAD is now at b361314 print
    # Testing reconstruction methods
    # on MNIST
    Hyperparameters:
    dataset=MNIST
    seed=0
    nn_lr=0.01
    nn_steps=100
    batch_size=128
    n_random_projections=512
    inv_lr=0.1
    inv_steps=100
    f_reg=0.0
    f_crit=1.0
    f_stats=0.001
    size_A=-1
    size_B=512
    show_after=20
    r_distort_level=0.5
    r_block_depth=4
    r_block_width=4
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=True
    scale_each=False
    reset_stats=False 
    
    Running on 'cuda'
    
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet20.pt
    net accuracy: 90.6%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet9.pt
    verifier net accuracy: 96.2%
    
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-+-RP-CC-512.pt.
    
    ground truth:



![png](/content/Thesis/figures/fig_18_1.png)


    
    
    distorted:



![png](/content/Thesis/figures/fig_18_3.png)


    
    
    
    
    ## Method: CRITERION
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_5.png)


    


     20%|██        |20.0/100 [00:24<01:39, 1.24s/epoch, accuracy=0.844, ideal=0.485, loss=0.685, psnr=12.9, |grad|=0.00649]

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_9.png)


    


     40%|████      |40.0/100 [00:49<01:14, 1.24s/epoch, accuracy=0.875, ideal=0.131, loss=0.462, psnr=8.38, |grad|=0.0125]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_13.png)


    


     60%|██████    |60.0/100 [01:14<00:49, 1.25s/epoch, accuracy=0.867, ideal=0.285, loss=0.361, psnr=7.67, |grad|=0.0309]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_17.png)


    


     80%|████████  |80.0/100 [01:39<00:24, 1.25s/epoch, accuracy=0.898, ideal=0.272, loss=0.306, psnr=7.4, |grad|=0.023] 

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_21.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.24s/epoch, accuracy=0.773, ideal=0.428, loss=0.901, psnr=9.01, |grad|=0.0072]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_25.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.773, ideal=0.428, loss=0.901, psnr=9.01, |grad|=0.0072]

    


    



![png](/content/Thesis/figures/fig_18_30.png)



![png](/content/Thesis/figures/fig_18_31.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_33.png)


    
    Results:
    	loss: 3.610
    	average PSNR: 7.664 | (distorted: 6.970)
    	rel. l2 reconstruction error: 14.270 | (distorted: 24.183)
    	nn accuracy: 67.8 %
    	nn validation set accuracy: 65.3 %
    	nn verifier accuracy: 16.8 %
    
    
    
    ## Method: NN
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_35.png)


    


     20%|██        |20.0/100 [00:24<01:38, 1.24s/epoch, accuracy=0.133, ideal=5.85, loss=23.3, psnr=10.3, |grad|=0.0103] 

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_39.png)


    


     40%|████      |40.0/100 [00:49<01:16, 1.27s/epoch, accuracy=0.156, ideal=2.48, loss=11.3, psnr=8.42, |grad|=0.0309]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_43.png)


    


     60%|██████    |60.0/100 [01:14<00:50, 1.25s/epoch, accuracy=0.219, ideal=2.46, loss=9.14, psnr=8.76, |grad|=0.0436]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_47.png)


    


     80%|████████  |80.0/100 [01:39<00:25, 1.25s/epoch, accuracy=0.164, ideal=4.83, loss=8.19, psnr=8.01, |grad|=0.165]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_51.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.445, ideal=5.39, loss=6.31, psnr=10.4, |grad|=0.0701]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_55.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.445, ideal=5.39, loss=6.31, psnr=10.4, |grad|=0.0701]

    


    



![png](/content/Thesis/figures/fig_18_60.png)



![png](/content/Thesis/figures/fig_18_61.png)



![png](/content/Thesis/figures/fig_18_62.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_64.png)


    
    Results:
    	loss: 27.875
    	average PSNR: 10.362 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.545 | (distorted: 24.183)
    	nn accuracy: 42.0 %
    	nn validation set accuracy: 41.2 %
    	nn verifier accuracy: 55.3 %
    
    
    
    ## Method: NN CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_66.png)


    


     20%|██        |20.0/100 [00:25<01:42, 1.29s/epoch, accuracy=0.438, ideal=2.28, loss=9.66, psnr=10.4, |grad|=0.0599]

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_70.png)


    


     40%|████      |40.0/100 [00:51<01:18, 1.31s/epoch, accuracy=0.742, ideal=1.66, loss=6.16, psnr=12.4, |grad|=0.279]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_74.png)


    


     60%|██████    |60.0/100 [01:17<00:51, 1.29s/epoch, accuracy=0.172, ideal=2.24, loss=14.9, psnr=9.12, |grad|=0.048]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_78.png)


    


     80%|████████  |80.0/100 [01:43<00:25, 1.28s/epoch, accuracy=0.711, ideal=2.43, loss=5.67, psnr=10.6, |grad|=0.0058]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_82.png)


    


    100%|██████████|100.0/100 [02:09<00:00, 1.28s/epoch, accuracy=0.742, ideal=2.52, loss=4.36, psnr=10.5, |grad|=0.305]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_86.png)


    


    100%|██████████|100.0/100 [02:09<00:00, 1.30s/epoch, accuracy=0.742, ideal=2.52, loss=4.36, psnr=10.5, |grad|=0.305]

    


    



![png](/content/Thesis/figures/fig_18_91.png)



![png](/content/Thesis/figures/fig_18_92.png)



![png](/content/Thesis/figures/fig_18_93.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_95.png)


    
    Results:
    	loss: 19.476
    	average PSNR: 10.392 | (distorted: 6.970)
    	rel. l2 reconstruction error: 13.238 | (distorted: 24.183)
    	nn accuracy: 75.0 %
    	nn validation set accuracy: 73.9 %
    	nn verifier accuracy: 52.9 %
    
    
    
    ## Method: NN ALL
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_97.png)


    


     20%|██        |20.0/100 [00:27<01:48, 1.36s/epoch, accuracy=0.844, ideal=1.09, loss=1.35, psnr=12.4, |grad|=0.0201]

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_101.png)


    


     40%|████      |40.0/100 [00:54<01:21, 1.35s/epoch, accuracy=0.938, ideal=0.509, loss=0.741, psnr=12.3, |grad|=0.011]  

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_105.png)


    


     60%|██████    |60.0/100 [01:21<00:55, 1.38s/epoch, accuracy=0.93, ideal=0.585, loss=0.789, psnr=11.7, |grad|=0.00891]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_109.png)


    


     80%|████████  |80.0/100 [01:49<00:27, 1.36s/epoch, accuracy=0.961, ideal=0.774, loss=0.707, psnr=12.3, |grad|=0.0326]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_113.png)


    


    100%|██████████|100.0/100 [02:16<00:00, 1.36s/epoch, accuracy=0.891, ideal=1.07, loss=0.876, psnr=11.9, |grad|=0.0545]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_117.png)


    


    100%|██████████|100.0/100 [02:16<00:00, 1.37s/epoch, accuracy=0.891, ideal=1.07, loss=0.876, psnr=11.9, |grad|=0.0545]

    


    



![png](/content/Thesis/figures/fig_18_122.png)



![png](/content/Thesis/figures/fig_18_123.png)



![png](/content/Thesis/figures/fig_18_124.png)



![png](/content/Thesis/figures/fig_18_125.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_127.png)


    
    Results:
    	loss: 3.164
    	average PSNR: 11.573 | (distorted: 6.970)
    	rel. l2 reconstruction error: 12.511 | (distorted: 24.183)
    	nn accuracy: 93.0 %
    	nn validation set accuracy: 88.5 %
    	nn verifier accuracy: 87.1 %
    
    
    
    ## Method: NN ALL CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_129.png)


    


     20%|██        |20.0/100 [00:47<03:10, 2.38s/epoch, accuracy=0.789, ideal=0.924, loss=2.15, psnr=13.2, |grad|=0.0406]

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_133.png)


    


     40%|████      |40.0/100 [01:35<02:22, 2.38s/epoch, accuracy=0.938, ideal=0.569, loss=1.21, psnr=11.7, |grad|=0.00674]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_137.png)


    


     60%|██████    |60.0/100 [02:23<01:35, 2.38s/epoch, accuracy=0.875, ideal=0.75, loss=1.04, psnr=11.5, |grad|=0.00996] 

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_141.png)


    


     80%|████████  |80.0/100 [03:11<00:47, 2.37s/epoch, accuracy=0.938, ideal=0.709, loss=0.885, psnr=11, |grad|=0.00259]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_145.png)


    


    100%|██████████|100.0/100 [03:58<00:00, 2.37s/epoch, accuracy=0.922, ideal=0.909, loss=1.07, psnr=11.2, |grad|=0.00539]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_149.png)


    


    100%|██████████|100.0/100 [03:58<00:00, 2.39s/epoch, accuracy=0.922, ideal=0.909, loss=1.07, psnr=11.2, |grad|=0.00539]

    


    



![png](/content/Thesis/figures/fig_18_154.png)



![png](/content/Thesis/figures/fig_18_155.png)



![png](/content/Thesis/figures/fig_18_156.png)



![png](/content/Thesis/figures/fig_18_157.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_159.png)


    
    Results:
    	loss: 4.273
    	average PSNR: 11.324 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.535 | (distorted: 24.183)
    	nn accuracy: 91.0 %
    	nn validation set accuracy: 87.7 %
    	nn verifier accuracy: 91.0 %
    
    
    
    ## Method: RP
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_161.png)


    


     20%|██        |20.0/100 [00:25<01:40, 1.26s/epoch, accuracy=0.844, ideal=0.489, loss=0.703, psnr=13, |grad|=0.00325] 

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_165.png)


    


     40%|████      |40.0/100 [00:51<01:17, 1.29s/epoch, accuracy=0.82, ideal=0.135, loss=0.613, psnr=8.33, |grad|=0.0526] 

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_169.png)


    


     60%|██████    |60.0/100 [01:16<00:50, 1.26s/epoch, accuracy=0.844, ideal=0.289, loss=0.478, psnr=7.78, |grad|=0.00754]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_173.png)


    


     80%|████████  |80.0/100 [01:42<00:25, 1.27s/epoch, accuracy=0.906, ideal=0.275, loss=0.328, psnr=9.39, |grad|=0.0246]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_177.png)


    


    100%|██████████|100.0/100 [02:07<00:00, 1.27s/epoch, accuracy=0.898, ideal=0.432, loss=0.353, psnr=9.65, |grad|=0.0208]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_181.png)


    


    100%|██████████|100.0/100 [02:08<00:00, 1.28s/epoch, accuracy=0.898, ideal=0.432, loss=0.353, psnr=9.65, |grad|=0.0208]

    


    



![png](/content/Thesis/figures/fig_18_186.png)



![png](/content/Thesis/figures/fig_18_187.png)



![png](/content/Thesis/figures/fig_18_188.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_190.png)


    
    Results:
    	loss: 1.098
    	average PSNR: 9.227 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.509 | (distorted: 24.183)
    	nn accuracy: 92.8 %
    	nn validation set accuracy: 85.3 %
    	nn verifier accuracy: 24.8 %
    
    
    
    ## Method: RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_192.png)


    


     20%|██        |20.0/100 [00:27<01:49, 1.37s/epoch, accuracy=0.844, ideal=0.495, loss=0.706, psnr=13, |grad|=0.00247]  

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_196.png)


    


     40%|████      |40.0/100 [00:54<01:21, 1.35s/epoch, accuracy=0.906, ideal=0.141, loss=0.252, psnr=10.8, |grad|=0.00897]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_200.png)


    


     60%|██████    |60.0/100 [01:21<00:54, 1.35s/epoch, accuracy=0.93, ideal=0.295, loss=0.187, psnr=9.46, |grad|=0.00641]

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_204.png)


    


     80%|████████  |80.0/100 [01:48<00:26, 1.34s/epoch, accuracy=0.969, ideal=0.281, loss=0.123, psnr=9.93, |grad|=0.00253]

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_208.png)


    


    100%|██████████|100.0/100 [02:15<00:00, 1.34s/epoch, accuracy=0.961, ideal=0.439, loss=0.147, psnr=9.12, |grad|=0.00315]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_212.png)


    


    100%|██████████|100.0/100 [02:15<00:00, 1.36s/epoch, accuracy=0.961, ideal=0.439, loss=0.147, psnr=9.12, |grad|=0.00315]

    


    



![png](/content/Thesis/figures/fig_18_217.png)



![png](/content/Thesis/figures/fig_18_218.png)



![png](/content/Thesis/figures/fig_18_219.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_221.png)


    
    Results:
    	loss: 0.693
    	average PSNR: 9.155 | (distorted: 6.970)
    	rel. l2 reconstruction error: 14.112 | (distorted: 24.183)
    	nn accuracy: 96.3 %
    	nn validation set accuracy: 90.2 %
    	nn verifier accuracy: 62.9 %
    
    
    
    ## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_18_223.png)


    


     20%|██        |20.0/100 [00:50<03:22, 2.53s/epoch, accuracy=0.812, ideal=0.905, loss=2.01, psnr=13.7, |grad|=0.0162]

    
    epoch 20:



![png](/content/Thesis/figures/fig_18_227.png)


    


     40%|████      |40.0/100 [01:41<02:30, 2.51s/epoch, accuracy=0.914, ideal=0.55, loss=1.18, psnr=13.2, |grad|=0.000701]

    
    epoch 40:



![png](/content/Thesis/figures/fig_18_231.png)


    


     60%|██████    |60.0/100 [02:31<01:40, 2.50s/epoch, accuracy=0.875, ideal=0.73, loss=1.1, psnr=12.6, |grad|=0.00875]  

    
    epoch 60:



![png](/content/Thesis/figures/fig_18_235.png)


    


     80%|████████  |80.0/100 [03:22<00:51, 2.57s/epoch, accuracy=0.93, ideal=0.69, loss=0.906, psnr=11.9, |grad|=0.00261] 

    
    epoch 80:



![png](/content/Thesis/figures/fig_18_239.png)


    


    100%|██████████|100.0/100 [04:12<00:00, 2.52s/epoch, accuracy=0.906, ideal=0.888, loss=0.958, psnr=10.8, |grad|=0.0145]

    
    epoch 100:



![png](/content/Thesis/figures/fig_18_243.png)


    


    100%|██████████|100.0/100 [04:12<00:00, 2.53s/epoch, accuracy=0.906, ideal=0.888, loss=0.958, psnr=10.8, |grad|=0.0145]

    


    



![png](/content/Thesis/figures/fig_18_248.png)



![png](/content/Thesis/figures/fig_18_249.png)



![png](/content/Thesis/figures/fig_18_250.png)



![png](/content/Thesis/figures/fig_18_251.png)


    Inverted:
    50 / 512 



![png](/content/Thesis/figures/fig_18_253.png)


    
    Results:
    	loss: 3.857
    	average PSNR: 10.484 | (distorted: 6.970)
    	rel. l2 reconstruction error: 13.790 | (distorted: 24.183)
    	nn accuracy: 92.0 %
    	nn validation set accuracy: 87.2 %
    	nn verifier accuracy: 89.1 %
    
    # Summary
    =========
    
    
    baseline       acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    -----------------------------------------------------------
    B (original)   0.89  0.90      0.96      --        --      
    B (distorted)  0.09  0.11      0.08      6.97      24.18   
    A              0.91  --        0.96      --        --      
    
    Reconstruction methods:
    
    method          acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    ------------------------------------------------------------
    CRITERION       0.68  0.65      0.17      7.66      14.27   
    NN              0.42  0.41      0.55      10.36     15.55   
    NN CC           0.75  0.74      0.53      10.39     13.24   
    NN ALL          0.93  0.88      0.87      11.57     12.51   
    NN ALL CC       0.91  0.88      0.91      11.32     15.53   
    RP              0.93  0.85      0.25      9.23      15.51   
    RP CC           0.96  0.90      0.63      9.16      14.11   
    NN ALL + RP CC  0.92  0.87      0.89      10.48     13.79   



```
# deeper blocks
!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=MNIST \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=-1 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=0.001 \
-r_distort_level=0.5 \
-r_block_depth=8 \
--plot_ideal \
-show_after=20 \
--reset_stats \
```

# INVERSION

nbconv:
    images: 'inversion_cifar10'

## CIFAR10


```
# USING SCALE_EACH
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=700 \
-batch_size=256 \
-size_A=1024 \
-size_B=256 \
-f_reg=0.001 \
-f_crit=1 \
-f_stats=100 \
--use_jitter \
--plot_ideal \
--scale_each \

```

    Already up to date.
    HEAD is now at 8b1cf4e depth
    # Testing reconstruction methods
    # on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=0
    nn_lr=0.01
    nn_steps=100
    batch_size=256
    n_random_projections=512
    inv_lr=0.1
    inv_steps=700
    f_reg=0.001
    f_crit=1.0
    f_stats=100.0
    size_A=1024
    size_B=256
    show_after=50
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=True
    plot_ideal=True
    scale_each=True 
    
    Running on 'cuda'
    
    Files already downloaded and verified
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.5%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 81.8%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.


    100%|██████████| 4/4 [00:00<00:00, 18.27batch/s]

    
    Saving data to /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-ReLU-512.pt.


    
    100%|██████████| 4/4 [00:00<00:00, 16.85batch/s]

    
    Saving data to /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-ReLU-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-512.pt.
    
    
    
    ## Method: CRITERION
    
    
    epoch 0:


    



![png](/content/Thesis/figures/fig_23_6.png)


    


      7%|▋         |50.0/700 [00:50<10:45,1.01epoch/s, accuracy=0.992, ideal=0.502, loss=0.568, |grad|=0.00687]

    
    epoch 50:



![png](/content/Thesis/figures/fig_23_10.png)


    


      9%|▉         |64.0/700 [01:04<10:30,1.01epoch/s, accuracy=1, ideal=0.502, loss=0.535, |grad|=0.00524]


```
# BASELINE, SANITY RUN

!git pull
!git reset --hard origin/master
%run ext/Nvlabs/cifar10/deepinversion-redo.py
```

    Already up to date.
    HEAD is now at 600a49c redo 700 iters
    loading resnet34
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=700.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_24_3.png)



![png](/content/Thesis/figures/fig_24_4.png)



```
# REDO MEAN + SCALE EACH
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=CIFAR10 \
-inv_lr=0.1 \
-inv_steps=700 \
-batch_size=256 \
-size_B=256 \
-f_reg=0.001 \
-f_crit=1 \
-f_stats=10 \
-n_random_projections=256 \
-show_after=100
--use_jitter \
--plot_ideal \
--scale_each \

```

    Already up to date.
    HEAD is now at d8ceb64 stats comparability
    # Testing reconstruction methods
    # on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=0
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=False
    nn_lr=0.01
    nn_steps=100
    batch_size=256
    n_random_projections=256
    inv_lr=0.1
    inv_steps=700
    f_reg=0.001
    f_crit=1.0
    f_stats=10.0
    size_A=-1
    size_B=256
    show_after=100 
    
    Running on 'cuda'
    
    Files already downloaded and verified
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.2%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 80.6%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-256.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-256.pt.
    
    
    ## Method: CRITERION
    


     14%|█▍        |101.0/700 [02:48<16:39,1.67s/epoch, accuracy=1, loss=0.581, |grad|=0.00437]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_3.png)


    


     29%|██▊       |201.0/700 [05:35<13:52,1.67s/epoch, accuracy=1, loss=0.385, |grad|=0.00545]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_7.png)


    


     43%|████▎     |301.0/700 [08:22<11:06,1.67s/epoch, accuracy=1, loss=0.374, |grad|=0.0067] 

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_11.png)


    


     57%|█████▋    |401.0/700 [11:09<08:18,1.67s/epoch, accuracy=1, loss=0.398, |grad|=0.00712]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_15.png)


    


     72%|███████▏  |501.0/700 [13:56<05:32,1.67s/epoch, accuracy=1, loss=0.376, |grad|=0.00629]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_19.png)


    


     86%|████████▌ |601.0/700 [16:43<02:45,1.67s/epoch, accuracy=1, loss=0.389, |grad|=0.00639]

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_23.png)


    


    100%|██████████|700.0/700 [19:29<00:00,1.67s/epoch, accuracy=1, loss=0.378, |grad|=0.00628]

    


    



![png](/content/Thesis/figures/fig_25_28.png)



![png](/content/Thesis/figures/fig_25_29.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_31.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 39.8 %
    
    
    ## Method: NN
    


     14%|█▍        |101.0/700 [02:48<16:42,1.67s/epoch, accuracy=1, loss=0.712, |grad|=0.00392]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_35.png)


    


     29%|██▊       |201.0/700 [05:36<13:54,1.67s/epoch, accuracy=1, loss=0.404, |grad|=0.00486]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_39.png)


    


     43%|████▎     |301.0/700 [08:23<11:07,1.67s/epoch, accuracy=1, loss=0.395, |grad|=0.00506]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_43.png)


    


     57%|█████▋    |401.0/700 [11:11<08:20,1.67s/epoch, accuracy=1, loss=0.389, |grad|=0.00523]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_47.png)


    


     72%|███████▏  |501.0/700 [13:58<05:32,1.67s/epoch, accuracy=1, loss=0.416, |grad|=0.00469]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_51.png)


    


     86%|████████▌ |601.0/700 [16:46<02:45,1.67s/epoch, accuracy=1, loss=0.392, |grad|=0.00559]

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_55.png)


    


    100%|██████████|700.0/700 [19:32<00:00,1.67s/epoch, accuracy=1, loss=0.4, |grad|=0.0115]

    


    



![png](/content/Thesis/figures/fig_25_60.png)



![png](/content/Thesis/figures/fig_25_61.png)



![png](/content/Thesis/figures/fig_25_62.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_64.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 41.0 %
    
    
    ## Method: NN CC
    


     14%|█▍        |101.0/700 [02:50<16:48,1.68s/epoch, accuracy=0.984, loss=1.17, |grad|=0.0916]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_68.png)


    


     29%|██▊       |201.0/700 [05:38<13:59,1.68s/epoch, accuracy=0.996, loss=1.11, |grad|=0.0611]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_72.png)


    


     43%|████▎     |301.0/700 [08:26<11:11,1.68s/epoch, accuracy=1, loss=0.953, |grad|=0.0137]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_76.png)


    


     57%|█████▋    |401.0/700 [11:15<08:22,1.68s/epoch, accuracy=0.996, loss=1.04, |grad|=0.042] 

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_80.png)


    


     72%|███████▏  |501.0/700 [14:03<05:34,1.68s/epoch, accuracy=1, loss=1.05, |grad|=0.0128]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_84.png)


    


     86%|████████▌ |601.0/700 [16:51<02:46,1.68s/epoch, accuracy=1, loss=0.936, |grad|=0.0161] 

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_88.png)


    


    100%|██████████|700.0/700 [19:38<00:00,1.68s/epoch, accuracy=0.996, loss=1.72, |grad|=0.016]

    


    



![png](/content/Thesis/figures/fig_25_93.png)



![png](/content/Thesis/figures/fig_25_94.png)



![png](/content/Thesis/figures/fig_25_95.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_97.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 64.1 %
    
    
    ## Method: NN ALL
    


     14%|█▍        |101.0/700 [02:53<17:11,1.72s/epoch, accuracy=0.992, loss=9.26, |grad|=0.0911]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_101.png)


    


     29%|██▊       |201.0/700 [05:46<14:19,1.72s/epoch, accuracy=0.992, loss=5.52, |grad|=0.0787]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_105.png)


    


     43%|████▎     |301.0/700 [08:38<11:27,1.72s/epoch, accuracy=0.996, loss=3.65, |grad|=0.0894]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_109.png)


    


     57%|█████▋    |401.0/700 [11:31<08:34,1.72s/epoch, accuracy=0.996, loss=2.82, |grad|=0.0825]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_113.png)


    


     72%|███████▏  |501.0/700 [14:23<05:42,1.72s/epoch, accuracy=0.996, loss=2.63, |grad|=0.0848]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_117.png)


    


     86%|████████▌ |601.0/700 [17:16<02:50,1.73s/epoch, accuracy=0.996, loss=2.46, |grad|=0.0807]

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_121.png)


    


    100%|██████████|700.0/700 [20:07<00:00,1.72s/epoch, accuracy=1, loss=2.84, |grad|=0.0759]

    


    



![png](/content/Thesis/figures/fig_25_126.png)



![png](/content/Thesis/figures/fig_25_127.png)



![png](/content/Thesis/figures/fig_25_128.png)



![png](/content/Thesis/figures/fig_25_129.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_131.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 75.4 %
    
    
    ## Method: NN ALL CC
    


     14%|█▍        |101.0/700 [03:29<20:44,2.08s/epoch, accuracy=1, loss=1.57, |grad|=0.00304]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_135.png)


    


     29%|██▊       |201.0/700 [06:58<17:19,2.08s/epoch, accuracy=1, loss=1.26, |grad|=0.00259]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_139.png)


    


     43%|████▎     |301.0/700 [10:26<13:51,2.08s/epoch, accuracy=1, loss=1.01, |grad|=0.00239]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_143.png)


    


     57%|█████▋    |401.0/700 [13:54<10:21,2.08s/epoch, accuracy=1, loss=0.753, |grad|=0.00262]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_147.png)


    


     72%|███████▏  |501.0/700 [17:22<06:53,2.08s/epoch, accuracy=1, loss=0.697, |grad|=0.00326]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_151.png)


    


     86%|████████▌ |601.0/700 [20:50<03:25,2.08s/epoch, accuracy=1, loss=0.669, |grad|=0.00443]

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_155.png)


    


    100%|██████████|700.0/700 [24:15<00:00,2.08s/epoch, accuracy=1, loss=0.653, |grad|=0.00409]

    


    



![png](/content/Thesis/figures/fig_25_160.png)



![png](/content/Thesis/figures/fig_25_161.png)



![png](/content/Thesis/figures/fig_25_162.png)



![png](/content/Thesis/figures/fig_25_163.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_165.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 41.8 %
    
    
    ## Method: RP
    


     14%|█▍        |101.0/700 [02:49<16:42,1.67s/epoch, accuracy=1, loss=4.28, |grad|=0.104]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_169.png)


    


     29%|██▊       |201.0/700 [05:36<13:54,1.67s/epoch, accuracy=1, loss=2.91, |grad|=0.105]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_173.png)


    


     43%|████▎     |301.0/700 [08:24<11:07,1.67s/epoch, accuracy=1, loss=2.53, |grad|=0.104]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_177.png)


    


     57%|█████▋    |401.0/700 [11:11<08:20,1.67s/epoch, accuracy=1, loss=2.35, |grad|=0.104]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_181.png)


    


     72%|███████▏  |501.0/700 [13:59<05:32,1.67s/epoch, accuracy=1, loss=2.31, |grad|=0.104]

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_185.png)


    


     86%|████████▌ |601.0/700 [16:46<02:45,1.67s/epoch, accuracy=1, loss=2.14, |grad|=0.104]

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_189.png)


    


    100%|██████████|700.0/700 [19:32<00:00,1.68s/epoch, accuracy=1, loss=2.19, |grad|=0.104]

    


    



![png](/content/Thesis/figures/fig_25_194.png)



![png](/content/Thesis/figures/fig_25_195.png)



![png](/content/Thesis/figures/fig_25_196.png)


    Inverted:



![png](/content/Thesis/figures/fig_25_198.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 35.2 %
    
    
    ## Method: RP CC
    


     14%|█▍        |101.0/700 [02:50<16:52,1.69s/epoch, accuracy=0.801, loss=13.2, |grad|=2.12]

    
    epoch 100:



![png](/content/Thesis/figures/fig_25_202.png)


    


     29%|██▊       |201.0/700 [05:39<14:03,1.69s/epoch, accuracy=0.891, loss=13.4, |grad|=2.13]

    
    epoch 200:



![png](/content/Thesis/figures/fig_25_206.png)


    


     43%|████▎     |301.0/700 [08:29<11:14,1.69s/epoch, accuracy=0.934, loss=12.3, |grad|=2.04]

    
    epoch 300:



![png](/content/Thesis/figures/fig_25_210.png)


    


     57%|█████▋    |401.0/700 [11:18<08:28,1.70s/epoch, accuracy=0.961, loss=13.2, |grad|=2.06]

    
    epoch 400:



![png](/content/Thesis/figures/fig_25_214.png)


    


     72%|███████▏  |501.0/700 [14:07<05:36,1.69s/epoch, accuracy=0.973, loss=11.2, |grad|=2]   

    
    epoch 500:



![png](/content/Thesis/figures/fig_25_218.png)


    


     86%|████████▌ |601.0/700 [16:56<02:47,1.69s/epoch, accuracy=0.984, loss=10.3, |grad|=1.9] 

    
    epoch 600:



![png](/content/Thesis/figures/fig_25_222.png)


    


     89%|████████▉ |625.0/700 [17:37<02:06,1.69s/epoch, accuracy=0.988, loss=9.98, |grad|=2.06]

nbconv:
    images: 'inversion_mnist'

## MNIST


```
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=MNIST \
-n_random_projections=512 \
-nn_steps=1 \
-size_A=-1 \
-size_B=128 \
-batch_size=128 \
-inv_lr=0.05 \
-inv_steps=500 \
-f_reg=0.0005 \
-f_crit=1 \
-f_stats=0.001 \
-show_after=100 \
-seed=-1 \
--use_jitter \
--plot_ideal \

# --nn_resume_train \
# --reset_stats \

```

    Already up to date.
    HEAD is now at efb7fd3 inversion upd
    # Testing reconstruction methods
    # on MNIST
    Hyperparameters:
    dataset=MNIST
    seed=-1
    nn_lr=0.01
    nn_steps=1
    batch_size=128
    n_random_projections=512
    inv_lr=0.05
    inv_steps=500
    f_reg=0.0005
    f_crit=1.0
    f_stats=0.001
    size_A=-1
    size_B=128
    show_after=100
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=True
    plot_ideal=True
    scale_each=False
    reset_stats=False 
    
    Running on 'cuda'
    
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet20.pt
    net accuracy: 99.4%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet9.pt
    verifier net accuracy: 95.9%
    
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-+-RP-CC-512.pt.
    
    
    
    ## Method: CRITERION
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_1.png)


    


     20%|██        |100.0/500 [00:15<01:00,6.65epoch/s, accuracy=0.961, ideal=0.058, loss=0.208, |grad|=0.00124]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_5.png)


    


     40%|████      |200.0/500 [00:30<00:45,6.63epoch/s, accuracy=0.945, ideal=0.058, loss=0.34, |grad|=0.00205]

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_9.png)


    


     60%|██████    |300.0/500 [00:45<00:30,6.66epoch/s, accuracy=0.93, ideal=0.058, loss=0.685, |grad|=0.00274]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_13.png)


    


     80%|████████  |400.0/500 [01:00<00:15,6.61epoch/s, accuracy=0.922, ideal=0.058, loss=1.37, |grad|=0.00335]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_17.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.73epoch/s, accuracy=0.922, ideal=0.058, loss=1.5, |grad|=0.00273] 

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_21.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.60epoch/s, accuracy=0.922, ideal=0.058, loss=1.5, |grad|=0.00273]

    


    



![png](/content/Thesis/figures/fig_28_26.png)



![png](/content/Thesis/figures/fig_28_27.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_29.png)


    
    	nn accuracy: 92.2 %
    	nn verifier accuracy: 21.9 %
    
    
    
    ## Method: NN
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_31.png)


    


     20%|██        |100.0/500 [00:15<01:00,6.59epoch/s, accuracy=0.938, ideal=31.3, loss=4.22, |grad|=0.0164]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_35.png)


    


     40%|████      |200.0/500 [00:30<00:44,6.68epoch/s, accuracy=0.945, ideal=31.3, loss=3.87, |grad|=0.0075]

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_39.png)


    


     60%|██████    |300.0/500 [00:45<00:29,6.69epoch/s, accuracy=0.945, ideal=31.3, loss=4.02, |grad|=0.00296]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_43.png)


    


     80%|████████  |400.0/500 [01:00<00:14,6.76epoch/s, accuracy=0.891, ideal=31.3, loss=4.48, |grad|=0.0185]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_47.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.66epoch/s, accuracy=0.875, ideal=31.3, loss=5.72, |grad|=0.0273]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_51.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.63epoch/s, accuracy=0.875, ideal=31.3, loss=5.72, |grad|=0.0273]

    


    



![png](/content/Thesis/figures/fig_28_56.png)



![png](/content/Thesis/figures/fig_28_57.png)



![png](/content/Thesis/figures/fig_28_58.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_60.png)


    
    	nn accuracy: 89.8 %
    	nn verifier accuracy: 18.8 %
    
    
    
    ## Method: NN CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_62.png)


    


     20%|██        |100.0/500 [00:15<01:02,6.36epoch/s, accuracy=0.969, ideal=22.7, loss=4.41, |grad|=0.0331]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_66.png)


    


     40%|████      |200.0/500 [00:31<00:47,6.36epoch/s, accuracy=0.906, ideal=22.7, loss=4.02, |grad|=0.0121]

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_70.png)


    


     60%|██████    |300.0/500 [00:47<00:31,6.33epoch/s, accuracy=0.961, ideal=22.7, loss=3.76, |grad|=0.0122]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_74.png)


    


     80%|████████  |400.0/500 [01:03<00:15,6.34epoch/s, accuracy=0.953, ideal=22.7, loss=4.12, |grad|=0.0236]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_78.png)


    


    100%|██████████|500.0/500 [01:19<00:00,6.26epoch/s, accuracy=0.938, ideal=22.7, loss=4.33, |grad|=0.0207]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_82.png)


    


    100%|██████████|500.0/500 [01:19<00:00,6.27epoch/s, accuracy=0.938, ideal=22.7, loss=4.33, |grad|=0.0207]

    


    



![png](/content/Thesis/figures/fig_28_87.png)



![png](/content/Thesis/figures/fig_28_88.png)



![png](/content/Thesis/figures/fig_28_89.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_91.png)


    
    	nn accuracy: 93.0 %
    	nn verifier accuracy: 18.0 %
    
    
    
    ## Method: NN ALL
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_93.png)


    


     20%|██        |100.0/500 [00:16<01:07,5.94epoch/s, accuracy=0.938, ideal=4.63, loss=1.86, |grad|=0.00254]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_97.png)


    


     40%|████      |200.0/500 [00:33<00:50,5.92epoch/s, accuracy=0.992, ideal=4.63, loss=1.45, |grad|=0.00173]

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_101.png)


    


     60%|██████    |300.0/500 [00:50<00:33,5.98epoch/s, accuracy=0.977, ideal=4.63, loss=1.44, |grad|=0.00135]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_105.png)


    


     80%|████████  |400.0/500 [01:07<00:16,5.89epoch/s, accuracy=0.977, ideal=4.63, loss=1.43, |grad|=0.00157]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_109.png)


    


    100%|██████████|500.0/500 [01:24<00:00,5.95epoch/s, accuracy=0.969, ideal=4.63, loss=1.49, |grad|=0.00403]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_113.png)


    


    100%|██████████|500.0/500 [01:24<00:00,5.89epoch/s, accuracy=0.969, ideal=4.63, loss=1.49, |grad|=0.00403]

    


    



![png](/content/Thesis/figures/fig_28_118.png)



![png](/content/Thesis/figures/fig_28_119.png)



![png](/content/Thesis/figures/fig_28_120.png)



![png](/content/Thesis/figures/fig_28_121.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_123.png)


    
    	nn accuracy: 98.4 %
    	nn verifier accuracy: 10.2 %
    
    
    
    ## Method: NN ALL CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_125.png)


    


     20%|██        |100.0/500 [00:34<02:18,2.88epoch/s, accuracy=0.977, ideal=4.21, loss=1.99, |grad|=0.00301]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_129.png)


    


     40%|████      |200.0/500 [01:09<01:45,2.85epoch/s, accuracy=0.984, ideal=4.21, loss=1.7, |grad|=0.00178] 

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_133.png)


    


     60%|██████    |300.0/500 [01:43<01:09,2.90epoch/s, accuracy=0.984, ideal=4.21, loss=1.62, |grad|=0.00185]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_137.png)


    


     80%|████████  |400.0/500 [02:18<00:34,2.87epoch/s, accuracy=1, ideal=4.21, loss=1.48, |grad|=0.00231]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_141.png)


    


    100%|██████████|500.0/500 [02:53<00:00,2.85epoch/s, accuracy=0.984, ideal=4.21, loss=1.47, |grad|=0.00171]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_145.png)


    


    100%|██████████|500.0/500 [02:53<00:00,2.88epoch/s, accuracy=0.984, ideal=4.21, loss=1.47, |grad|=0.00171]

    


    



![png](/content/Thesis/figures/fig_28_150.png)



![png](/content/Thesis/figures/fig_28_151.png)



![png](/content/Thesis/figures/fig_28_152.png)



![png](/content/Thesis/figures/fig_28_153.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_155.png)


    
    	nn accuracy: 100.0 %
    	nn verifier accuracy: 14.8 %
    
    
    
    ## Method: RP
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_157.png)


    


     20%|██        |100.0/500 [00:14<00:59,6.72epoch/s, accuracy=0.953, ideal=0.0616, loss=0.231, |grad|=0.00186]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_161.png)


    


     40%|████      |200.0/500 [00:29<00:44,6.67epoch/s, accuracy=0.953, ideal=0.0616, loss=0.447, |grad|=0.00229]

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_165.png)


    


     60%|██████    |300.0/500 [00:45<00:29,6.75epoch/s, accuracy=0.938, ideal=0.0616, loss=0.746, |grad|=0.00239]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_169.png)


    


     80%|████████  |400.0/500 [00:59<00:14,6.79epoch/s, accuracy=0.93, ideal=0.0616, loss=1.02, |grad|=0.00219] 

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_173.png)


    


    100%|██████████|500.0/500 [01:14<00:00,6.78epoch/s, accuracy=0.922, ideal=0.0616, loss=1.48, |grad|=0.00268]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_177.png)


    


    100%|██████████|500.0/500 [01:14<00:00,6.67epoch/s, accuracy=0.922, ideal=0.0616, loss=1.48, |grad|=0.00268]

    


    



![png](/content/Thesis/figures/fig_28_182.png)



![png](/content/Thesis/figures/fig_28_183.png)



![png](/content/Thesis/figures/fig_28_184.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_186.png)


    
    	nn accuracy: 91.4 %
    	nn verifier accuracy: 23.4 %
    
    
    
    ## Method: RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_188.png)


    


     20%|██        |100.0/500 [00:16<01:06,6.06epoch/s, accuracy=0.953, ideal=0.0679, loss=0.221, |grad|=0.00142]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_192.png)


    


     40%|████      |200.0/500 [00:33<00:49,6.10epoch/s, accuracy=0.961, ideal=0.0679, loss=0.289, |grad|=0.002]  

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_196.png)


    


     60%|██████    |300.0/500 [00:49<00:32,6.07epoch/s, accuracy=0.938, ideal=0.0679, loss=0.682, |grad|=0.00248]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_200.png)


    


     80%|████████  |400.0/500 [01:06<00:16,6.03epoch/s, accuracy=0.922, ideal=0.0679, loss=1.07, |grad|=0.00269]

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_204.png)


    


    100%|██████████|500.0/500 [01:23<00:00,6.11epoch/s, accuracy=0.914, ideal=0.0679, loss=1.65, |grad|=0.00284]

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_208.png)


    


    100%|██████████|500.0/500 [01:23<00:00,6.00epoch/s, accuracy=0.914, ideal=0.0679, loss=1.65, |grad|=0.00284]

    


    



![png](/content/Thesis/figures/fig_28_213.png)



![png](/content/Thesis/figures/fig_28_214.png)



![png](/content/Thesis/figures/fig_28_215.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_217.png)


    
    	nn accuracy: 92.2 %
    	nn verifier accuracy: 23.4 %
    
    
    
    ## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](/content/Thesis/figures/fig_28_219.png)


    


     20%|██        |100.0/500 [00:36<02:25,2.76epoch/s, accuracy=0.953, ideal=4.03, loss=1.91, |grad|=0.00159]

    
    epoch 100:



![png](/content/Thesis/figures/fig_28_223.png)


    


     40%|████      |200.0/500 [01:12<01:49,2.74epoch/s, accuracy=1, ideal=4.03, loss=1.6, |grad|=0.00114] 

    
    epoch 200:



![png](/content/Thesis/figures/fig_28_227.png)


    


     60%|██████    |300.0/500 [01:48<01:12,2.77epoch/s, accuracy=0.992, ideal=4.03, loss=1.52, |grad|=0.00152]

    
    epoch 300:



![png](/content/Thesis/figures/fig_28_231.png)


    


     80%|████████  |400.0/500 [02:25<00:37,2.66epoch/s, accuracy=0.992, ideal=4.03, loss=1.46, |grad|=0.0019] 

    
    epoch 400:



![png](/content/Thesis/figures/fig_28_235.png)


    


    100%|██████████|500.0/500 [03:01<00:00,2.76epoch/s, accuracy=1, ideal=4.03, loss=1.39, |grad|=0.00178]   

    
    epoch 500:



![png](/content/Thesis/figures/fig_28_239.png)


    


    100%|██████████|500.0/500 [03:01<00:00,2.75epoch/s, accuracy=1, ideal=4.03, loss=1.39, |grad|=0.00178]

    


    



![png](/content/Thesis/figures/fig_28_244.png)



![png](/content/Thesis/figures/fig_28_245.png)



![png](/content/Thesis/figures/fig_28_246.png)



![png](/content/Thesis/figures/fig_28_247.png)


    Inverted:



![png](/content/Thesis/figures/fig_28_249.png)


    
    	nn accuracy: 99.2 %
    	nn verifier accuracy: 18.8 %
    
    # Summary
    =========
    
    
    method          acc   acc(ver)  
    --------------------------------
    CRITERION       0.92  0.22      
    NN              0.90  0.19      
    NN CC           0.93  0.18      
    NN ALL          0.98  0.10      
    NN ALL CC       1.00  0.15      
    RP              0.91  0.23      
    RP CC           0.92  0.23      
    NN ALL + RP CC  0.99  0.19      


nbconv:
    out: 'Tests'

# ->END
# GMMs



```
# !git pull
# !git reset --hard origin/master
%run projectOK/main.py \
-dataset=GMM \
-distort_level=1.0 \
-batch_size=-1 \
-nn_lr=0.01 \
-nn_steps=500 \
-n_random_projections=256 \
-inv_lr=0.1 \
-inv_steps=1000 \
-g_modes=12 \
-g_scale_mean=3 \
-g_scale_cov=20 \
-g_mean_shift=2.5 \
--nn_reset_train \
--use_drive \

# %run projectOK/mainGMM.py \
# -seed=300 \
# -n_classes=10 \
# -n_dims=20 \
# -n_samples_A=5000 \
# -n_samples_B=1000 \
# -n_samples_valid=5000 \
# -distort_level=1.0 \
# -g_modes=12 \
# -g_scale_mean=3 \
# -g_scale_cov=20 \
# -g_mean_shift=2.5 \
# -nn_lr=0.01 \
# -nn_steps=500 \
# -nn_width=32 \
# -nn_depth=4 \
# -n_random_projections=128 \
# -inv_lr=0.1 \
# -inv_steps=1000 \
# --nn_verifier \

```

    # Testing reconstruction methods
    # on GMM
    Hyperparameters:
    dataset=GMM
    seed=0
    nn_resume_train=False
    nn_reset_train=True
    use_amp=False
    use_drive=True
    use_var=False
    perturb_strength=1.0
    nn_lr=0.01
    nn_steps=500
    batch_size=-1
    n_random_projections=256
    inv_lr=0.1
    inv_steps=1000
    g_modes=12
    g_scale_mean=3.0
    g_scale_cov=20.0
    g_mean_shift=2.5
    
    Running on 'cuda'
    No Checkpoint found / Reset.
    Path: /content/drive/My Drive/Thesis/models/GMM/net_GMM_20-32-32-32-32-3
    Beginning training.


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



    HBox(children=(FloatProgress(value=0.0, max=500.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_4.png)


    No Checkpoint found / Reset.
    Path: /content/drive/My Drive/Thesis/models/GMM/net_GMM_20-32-32-32-32-3_verifier
    Beginning training.



    HBox(children=(FloatProgress(value=0.0, max=500.0), HTML(value='')))


    
    Loading stats from /content/drive/My Drive/Thesis/models/GMM/stats_inputs.pt.
    Loading stats from /content/drive/My Drive/Thesis/models/GMM/stats_inputs-CC.pt.
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_NN.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_NN-CC.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_NN-ALL.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_NN-ALL-CC.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_RP.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_RP-CC.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_RP-ReLU.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_RP-ReLU-CC.pt.
    
    Beginning tracking stats.



    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    Saving stats in /content/Thesis/models/GMM/stats_combined.pt.
    
    
    ## Method: NN
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_28.png)


    Results:
    	loss: 0.000
    	rel. l2 reconstruction error: 7.924
    	nn accuracy: 37.4 %
    	nn validation set accuracy: 40.4 %
    	nn verifier accuracy: 29.3 %
    
    ## Method: NN-CC
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_32.png)


    Results:
    	loss: 0.019
    	rel. l2 reconstruction error: 8.403
    	nn accuracy: 98.0 %
    	nn validation set accuracy: 68.7 %
    	nn verifier accuracy: 49.5 %
    
    ## Method: NN-ALL
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_36.png)


    Results:
    	loss: 0.002
    	rel. l2 reconstruction error: 2.717
    	nn accuracy: 32.3 %
    	nn validation set accuracy: 44.4 %
    	nn verifier accuracy: 33.3 %
    
    ## Method: NN-ALL-CC
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_40.png)


    Results:
    	loss: 0.067
    	rel. l2 reconstruction error: 6.580
    	nn accuracy: 98.0 %
    	nn validation set accuracy: 80.8 %
    	nn verifier accuracy: 86.9 %
    
    ## Method: RP
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_44.png)


    Results:
    	loss: 0.059
    	rel. l2 reconstruction error: 3.718
    	nn accuracy: 24.2 %
    	nn validation set accuracy: 28.3 %
    	nn verifier accuracy: 22.2 %
    
    ## Method: RP-CC
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_48.png)


    Results:
    	loss: 0.109
    	rel. l2 reconstruction error: 5.292
    	nn accuracy: 82.8 %
    	nn validation set accuracy: 70.7 %
    	nn verifier accuracy: 82.8 %
    
    ## Method: RP-ReLU
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_52.png)


    Results:
    	loss: 0.010
    	rel. l2 reconstruction error: 3.106
    	nn accuracy: 37.4 %
    	nn validation set accuracy: 32.3 %
    	nn verifier accuracy: 38.4 %
    
    ## Method: RP-ReLU-CC
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_56.png)


    Results:
    	loss: 0.062
    	rel. l2 reconstruction error: 5.377
    	nn accuracy: 80.8 %
    	nn validation set accuracy: 74.7 %
    	nn verifier accuracy: 79.8 %
    
    ## Method: combined
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))


    



![png](/content/Thesis/figures/fig_31_60.png)


    Results:
    	loss: 0.142
    	rel. l2 reconstruction error: 6.774
    	nn accuracy: 98.0 %
    	nn validation set accuracy: 85.9 %
    	nn verifier accuracy: 90.9 %
    
    # Summary
    =========
    
    
    baseline       acc   acc(val)  acc(ver)  
    -----------------------------------------
    B (original)   0.97  0.93      0.97      
    B (perturbed)  0.39  0.34      0.39      
    A              1.00  N.A.      1.00      
    
    Reconstruction methods:
    
    method      acc   acc(val)  acc(ver)  l2-err  loss  
    ----------------------------------------------------
    NN          0.37  0.40      0.29      7.92    0.00  
    NN-CC       0.98  0.69      0.49      8.40    0.02  
    NN-ALL      0.32  0.44      0.33      2.72    0.00  
    NN-ALL-CC   0.98  0.81      0.87      6.58    0.07  
    RP          0.24  0.28      0.22      3.72    0.06  
    RP-CC       0.83  0.71      0.83      5.29    0.11  
    RP-ReLU     0.37  0.32      0.38      3.11    0.01  
    RP-ReLU-CC  0.81  0.75      0.80      5.38    0.06  
    combined    0.98  0.86      0.91      6.77    0.14  



```
baseline       acc   acc(val)  c-entr  acc(ver)  
-------------------------------------------------
B (original)   0.91  0.92      54.99   0.91      
B (perturbed)  0.17  0.16      inf     0.16      
A              0.94  N.A.      54.99   0.94      

Reconstruction methods:

method      acc   acc(val)  acc(ver)  l2-err  loss  c-entr  
------------------------------------------------------------
NN          0.14  0.14      0.14      8.67    0.00  inf     
NN CC       0.89  0.88      0.87      5.46    0.14  inf     
NN ALL      0.14  0.14      0.13      3.12    0.00  inf     
NN ALL CC   0.90  0.90      0.89      0.93    0.04  58.37   
RP          0.11  0.10      0.10      2.90    0.00  inf     
RP CC       0.89  0.89      0.89      1.30    0.04  58.86   
RP ReLU     0.10  0.10      0.11      4.40    0.00  inf     
RP ReLU CC  0.90  0.89      0.89      1.13    0.02  inf     
combined    0.90  0.90      0.90      0.71    0.03  inf     
```



# TESTS

## Test 1


```
%run projectOK/test1.py
```

    Simple regression test for recovering an affine transformation..
    



![png](/content/Thesis/figures/fig_36_1.png)


    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', max=200.0, style=ProgressStyle(description_width='…


    



![png](/content/Thesis/figures/fig_36_5.png)


    After Pre-Processing:



![png](/content/Thesis/figures/fig_36_7.png)


    l2 reconstruction error: 0.886


## Test 2


```
%run projectOK/test2.py
```

    ERROR:root:File `'projectOK/test2.py'` not found.


## Test 3


```
%run projectOK/test3.py
```

    Testing reconstruction by matching statistics
    of Gaussian Mixture Model on random projections..
    
    Before:
    Cross Entropy of A: 3.711257219314575
    Cross Entropy of B: 1088.7333984375


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](/content/Thesis/figures/fig_40_2.png)



![png](/content/Thesis/figures/fig_40_3.png)


    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', max=400.0, style=ProgressStyle(description_width='…


    



![png](/content/Thesis/figures/fig_40_7.png)


    After Pre-Processing:
    Cross Entropy of B: 4.431112289428711
    Cross Entropy of unperturbed B: 3.6967499256134033



![png](/content/Thesis/figures/fig_40_9.png)


    l2 reconstruction error: 7.680



```
%run projectOK/test3bigger.py
```

    Test3 using random projections
    on 2 distinct clusters
    Comment:
    Method struggles with more seperated clusters
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](/content/Thesis/figures/fig_41_2.png)



![png](/content/Thesis/figures/fig_41_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', max=600.0, style=ProgressStyle(description_width='…


    



![png](/content/Thesis/figures/fig_41_7.png)


    After Pre-Processing:
    Cross Entropy of B: inf
    Cross Entropy of unperturbed B: 4.961302757263184



![png](/content/Thesis/figures/fig_41_9.png)



![png](/content/Thesis/figures/fig_41_10.png)



![png](/content/Thesis/figures/fig_41_11.png)


    l2 reconstruction error: 35.574


## Test 4


```
%run projectOK/test4.py
```

    Testing reconstruction by matching
    class-conditional statistics
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](/content/Thesis/figures/fig_43_2.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', style=ProgressStyle(description_width='initial')),…


    



![png](/content/Thesis/figures/fig_43_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.816658973693848
    Cross Entropy of unperturbed B: 4.961302757263184



![png](/content/Thesis/figures/fig_43_8.png)


    l2 reconstruction error: 1.083


## Test 5


```
%run projectOK/test5.py
```

    Testing reconstruction by matching
    class-conditional statistics on random projections
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](/content/Thesis/figures/fig_45_2.png)



![png](/content/Thesis/figures/fig_45_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', max=400.0, style=ProgressStyle(description_width='…


    



![png](/content/Thesis/figures/fig_45_7.png)


    After Pre-Processing:
    Cross Entropy of B: 6.226280689239502
    Cross Entropy of unperturbed B: 4.961302757263184



![png](/content/Thesis/figures/fig_45_9.png)


    l2 reconstruction error: 1.945


## Test 6


```
%run projectOK/test6.py
```

    Testing reconstruction by matching
    statistics on neural network feature
    
    Training Checkpoint restored: /content/Thesis/models/net_GMM_2-16-16-16-3.pt


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](/content/Thesis/figures/fig_47_2.png)


    Before:
    Cross Entropy of A: 5.5779805183410645
    Cross Entropy of B: inf
    Beginning Inversion.



    HBox(children=(FloatProgress(value=0.0, description='Step', max=400.0, style=ProgressStyle(description_width='…


    



![png](/content/Thesis/figures/fig_47_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.1775803565979



![png](/content/Thesis/figures/fig_47_8.png)

