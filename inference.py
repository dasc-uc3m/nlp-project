import requests
import html
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    headers = {"Content-Type": "application/json"}
    data = {"prompt": args.prompt}
    response = requests.post("http://localhost:5000/generate", headers=headers, json=data)
    answer = response.json()["response"]
    print(html.unescape(answer))

if __name__=="__main__":
    main()