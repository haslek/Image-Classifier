def validate_on_test(testloader,model,device):
    model.eval()
    correct = 0
    wrong = 0
    accuracy = 0
    model.to(device)
    with torch.no_grad():
        for image,label in iter(testloader):
            image,label = image.to(device),label.to(device)
            output = model.forward(image)
            prediction = torch.exp(output)
            is_correct = (label.data == prediction.max(dim=1)[1])
            correct = is_correct.sum().item()
            accuracy += is_correct.type(torch.FloatTensor).mean()
            
        wrong = len(testloader) - correct
    print('Accuracy is: {},predicted {} correctly. {} wrong.'.format(accuracy/len(testloader),correct,wrong))