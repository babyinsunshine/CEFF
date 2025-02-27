import torch
from .base_model import BaseModel
from .Encoder import init_Encoder, InfoNCE_Loss

class EncoderModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options and rewrite default values for existing options.

        """
        return parser

    def __init__(self, opt):
        """Initialize the encoder model.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        # Specify the training losses to be printed
        self.loss_names = ['InfoNCE']
        # Specify the models to be saved to disk
        if self.isTrain:
            self.model_names = ['Encoder']
        else:
            self.model_names = ['Encoder']

        # Define the encoder network
        self.netEncoder = init_Encoder(gpu_ids=self.gpu_ids)

        if self.isTrain:
            # Define the loss function
            self.criterion_InfoNCE = InfoNCE_Loss(temperature=0.07).to(self.device)
            self.optimizer = torch.optim.Adam(self.netEncoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary preprocessing steps.

        Parameters:
            input (dict): contains the data itself and its metadata information.
        """
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)

    def forward(self):
        """Forward pass"""
        self.z1, self.z2 = self.netEncoder(self.A, self.B)

    def backward(self):
        """Compute the loss and perform backpropagation"""
        # Compute the InfoNCE loss
        self.loss_InfoNCE = self.criterion_InfoNCE(self.z1, self.z2)
        self.loss = self.loss_InfoNCE
        self.loss.backward()

    def optimize_parameters(self):
        """Optimize model parameters"""
        self.forward()  # Forward pass
        self.optimizer.zero_grad()  # Clear gradients
        self.backward()  # Backpropagation
        self.optimizer.step()  # Update parameters