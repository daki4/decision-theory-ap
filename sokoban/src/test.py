class A:
    pass

class B:
    pass

def main(obj):
    obj = type(obj)
    if obj == str:
        print('str')
    if obj == int:
        print('int')
    if obj == A or obj == B:
        print('a or b')



