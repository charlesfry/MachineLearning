import torch
import torchvision

def main():
    x = torch.tensor((2,3))

    print(x:=x.new_ones(5,3))

    x = torch.randn_like(x,dtype=torch.float)
    print(x)

    torch.cuda.set_enabled_lms(True)

if __name__ == '__main__' :
         main()