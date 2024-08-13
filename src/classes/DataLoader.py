

class DataLoader:
    def __init__(self, subset: str):
        if subset == "train":
            self.data = ""

    def load(self):
        return self.data


    def show_representation(self):
        print(self.data)

    