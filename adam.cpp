#include <vector>
#include <cmath>

class AdamOptimizer {

    private:
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        int t;
        std::vector<double> m; //first moment vector
        std::vector<double> v; //second moment vector

    public:
        AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8):
            learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}  //hyperparameter initialization

        void initialize(size_t num_params) {

            m =std::vector<double>(num_params, 0.0);
            v =std::vector<double>(num_params, 0.0);
            t = 0;

        } //state vector intialization

        std::vector<double> update(const std::vector<double> & params,
        const std::vector<double> &gradients) {

            if ( t == 0) {

                initialize(params.size());

            }
            t++;

            std::vector<double> updated_params(params.size()); //new vector for updated parameters

            for (size_t i = 0; i < params.size(); i++) {

                m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]; //weighted average of past gradients
            }

            for (size_t i = 0; i < params.size(); i++) {

                v[i] = beta2 * v[i] + (1-beta[2]) * gradients[i] * gradients[i]; //weighted average of past squared gradients
                
            }

            double beta1_correction = 1.0 - std::pow(beta1, t); 

            std::vector<double> m_hat(params.size()); //m_hat vector init
            for (size_t i = 0; i < params.size(); i ++) {

                m_hat[i] = m[i] / beta1_correction; 

            }

            double beta2_correction = 1.0 - std::pow(beta2, t);
            std::vector<double> v_hat(params.size()); //v_hat vector init

            for (size_t i = 0; i < params.size(); i++) {

                v_hat[i] = v[i] /beta2_correction;

            }

            for(size_t i = 0; i < params.size(); i ++) {

                updated_params[i] = params[i] - learning_rate * m_hat[i] / (std::sqrt(v_hat[i]) + epsilon);

            }

            return updated_params;
            
    
        }

        double get_learning_rate() const {
            return learning_rate; }

        double get_beta1() const {
            return beta1;
        }
        
        double get_beta2() const {
            return beta2;
        }

        double get_epsilon() const {

            return epsilon;

        }

        int get_timstep() const {
            return t;
        }

};

/*
    notes on convergence analysis:
    regret: sum of all prev differences between f(theta_t) (stepwise prediction) and f(theta_*) (best solution in hindsight)

    convex loss functions: (f(y) - f(x)) / (y-x) >= f'(x) always stays above the gradient

    gradient upper bound : ||gt||_2 <= G ||gt||_inf <= G_inf

    lemma 10.2: sum sqrt(gt,i^2/t) <= 2G_inf ||g1:T,i||_2  (proof by induction)

    lemma 10.4: sum m_hat^2/sqrt(t*v_hat) <= 2/(1- gamma) (1/sqrt(1 - beta2)||g1:t||_2)

    rearrange sum m_hat^2/sqrt(t*v_hat) using algorithm 1 and apply inequality sqrt(1-b2^t)/(1-b1^t)^2 <= 1/(1-b1)^2
        

    side notes on convergence: 

            uniform convergence (same N for all x) > locally uniform convergence (same N for a compact susbet of x) 
            > pointwise convergence (fn->f for each x individually) 
            > almost uniform convergence (uniform except on a set of arbitrary small measure)
            > almost everywhere convergence (point wise except on measure zero sets)
        

 



*/