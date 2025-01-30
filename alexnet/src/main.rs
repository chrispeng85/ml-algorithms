use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct AlexNet {

    features: nn::Sequential,
    classifier: nn::Sequential, 


impl AlexNet {

    pub fn new(vs: &nn:Path, num_classes: i64) -> Self {

            let features == nn::seq()
                //conv1
                .add(nn::conv2d(
                    vs,
                    3,  //input channel RGB
                    96, //output channel
                    11, //kernel size, large scale features
                    nn::ConvConfig { //additional params
                        stride: 4, 
                        padding: 2,
                        ..Default::default() //default for other params

                    },       
                ))
                .add_fn(|xs| xs.relu())  //closure for relu, kind of like lambda function
                .add(nn::max_pool2d(3,2,0,1,true)) 
                .add_fn(|xs|, xs.relu())
                .add(nn::max_pool2d(3,2,0,1,true))
                .add_fn(|xs|, xs.shallow_clone().local_response_norm(5, 1e-4, 0.75, 1.0))

                //conv2
                .add(nn::conv2d(
                    vs,
                    96,
                    256,
                    5,nn::ConvConfig {
                        padding:2 ,
                        groups: 2,
                        ..Default::default()

                    },
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::max_pool2d(3,2,0,1,true))
                .add_fn(|xs|, xs.shallow_clone().local_response_norm(5,1e-4, 0.75, 1.0)) //local response normalization 

    }           //conv3
                .add(n::conv2d(
                    
                    vs,
                    256,
                    384,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        ..Default::default()
                    },

                ))
                .add_fn(|xs| xs.relu())

                //conv4
                .add(nn::conv2d(
                    vs,
                    384,
                    384,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        groups: 2,
                        ..Default::default()

                    },


                ))
                .add_fn(|xs| xs.relu())

                //conv5
                .add(nn::conv2d(
                    vs,
                    384,
                    256,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        groups: 2,
                        ..Default::default()
                    },

                ))
                .add_fn(|xs| xs.relu())
                .add(nn::max_pool2d(3,2,0,1, true));

            let classifier = nn::seq()
            .add_fn(|xs| xs.flat_view()) //flatten tensor to 1d vector
            //fc6
            .add(nn::linear(

                vs,
                256*6*6,
                4096,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.dropout(0.5, true)) //drop out to prevent overfitting
            //fc7
            .add(nn::linear(
                vs,
                4096,
                4096,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.dropout(0.5, true))
            //fc8
            .add(nn::linear(
                vs,
                4096,
                num_classes,
                Default::default(),
            ));

            Self {
                features: features
                classifier: classifier

            }
}

}

impl Module for AlexNet {

    fn forward(&self, xs: & Tensor) -> Tensor {

        let features = self.features.forward(xs);
        self.classifier.forard(&features)

    }

}


fn main() -> Result<(), Box<dyn std::error::Error>> {

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let net = AlexNet::new(&vs.root(), 1000);

    let batch_size = 1;
    let input = Tensor::zeros(&[batch_size, 3, 227, 227], (tch::Kind::Float, tch::Device::Cpu));

    let output = net.forward(&input);
    println!("output shape: {:?}", output.size());

    Ok(())

}



/*

    AlexNet: feature extraction + classification

    input: 227*227*3
    
    formula: output_size = ((input_size - kernel_size + 2 * padding)/ stride) + 1

    96 output channels (heuristics)

    max_pooling : dimensional reduction, feature hierarchy (combines small features into large features)

    8 learned layers:   5 convolutional + 3 fully-connected

    relu: f(x) = max(0, x) trains faster than tanh or sigmoid


    response normalized activity:
    b(x,y) = a(x,y) / ( k + \alpha sum (ax,y) ^2)^beta

        k,n,alpha, beta are hyper-parameters 
        a is input activity
        b is normalized output

        --multiple nearby strong features are suppressed 
        --sparse strong channels are in turn encouraged
        --encourages specialization









*/