class Asd:
    def __init__(self):
        self.val1 = None
        self.val2 = None

    def print(self):
        print(self.val1, self.val2)

    def init(self):
        asd = {
            self.val1: 5,
            self.val2: 6,
        }
        print (self.val1)
        return asd

asd = Asd()

asd.print()
print(asd.init())
asd.print()