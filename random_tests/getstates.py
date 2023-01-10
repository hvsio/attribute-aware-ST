import json

ds = ['train', 'test', 'dev']
state_list = set()
pref_list = set()

for dataset in ds:
    print(dataset)
    with open(f"./speechBSD/{dataset}.json") as f:
        f = f.read()
        j = json.loads(f)
        for i in j:
            state_list.add("<" + i['en_spk_state'] + ">")
            pref_list.add("<" + i['ja_spk_prefecture'] + ">" )

print(state_list)
print(pref_list)