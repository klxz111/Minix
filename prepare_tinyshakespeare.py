from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    out_dir = Path("data/tinyshakespeare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "input.txt"

    if out_file.exists():
        print(f"[skip] exists: {out_file}")
        return

    print(f"[download] {URL}")
    urlretrieve(URL, out_file)
    print(f"[done] saved to: {out_file}")


if __name__ == "__main__":
    main()