model=CNN()

#save in direct path
path='torch_model.pth'
torch.save(model,' path')

#load
model_load=torch.load(path)
model_load

#save weights of model
checkpoint={'state' : model.state_dict(), 'epoch':2 , 'loss':100}
torch.save(checkpoint,path)

model.load_state_dict(torch.load(path)['state'])

torch.load(path)['loss']
torch.load(path)['epoch']
