import json

if __name__ == "__main__":
    files = ['train', 'test', 'dev']
    l = set()
    for set in files:
        l.add(set)
        with open(f'./speechBSD/{set}.json') as f:
            f = f.read()
            f = json.loads(f)
            for i, entry in enumerate(f):
                l.add(f'{entry["en_speaker"]} - {entry["en_spk_gender"]}')


    with open('./gender_checker.txt', 'w') as f:
        f.write("\n".join(l))