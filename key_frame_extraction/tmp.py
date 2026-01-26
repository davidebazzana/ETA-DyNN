import json

with open('dataset.json', 'r') as f:
    dataset = json.load(f)

dataset_2024 = {
    "train": {
        "free": [],
        "infested": []
    },
    "val": {
        "free": [],
        "infested": []
    }
}

for partition in dataset:
    print("Partition:", partition)
    for infestation_class in dataset[partition]:
        print("Class:", infestation_class)
        for video in dataset[partition][infestation_class]:
            dataset_2024[partition][infestation_class].append({
                "label": video["label"],
                "id": video["id"]
            })

with open("dataset_2024.json", "w") as f:
    json.dump(dataset_2024, f)
