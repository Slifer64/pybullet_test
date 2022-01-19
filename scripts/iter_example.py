class MyList:

    def __init__(self, *numbers):

        self.__list = [num for num in numbers]

    def __iter__(self):

        self.ind = 0
        return self

    def __next__(self):
        if self.ind < len(self.__list):
            self.ind += 1
            return self.__list[self.ind - 1]
        else:
            raise StopIteration


if __name__ == '__main__':

    # my_list = [i**2 for i in range(10)]

    my_list = MyList(*[i ** 2 for i in range(10)])

    it = iter(my_list)
    try:
        while True:
            print(next(it))
    except StopIteration:
        pass
    finally:
        del it

    it = iter(my_list)
    while True:
        try:
            print(next(it))
        except StopIteration:
            break
