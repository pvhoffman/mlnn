#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <armadillo>
#include <vector>
#include <string>
#include <stdexcept>

namespace mlnn
{
//---------------------------------------------------------
class NNetwork
{
    public:  //types
        static const unsigned nprediction = (unsigned)-1;

        struct NNLayerDesc_t {
                // number of a/z units in layer *NOT* counting the bias
                unsigned unitCount;
        };
    private:
        unsigned _classCount;

        double _lambda;

        // theta values
        std::vector< arma::mat > _ts;

        // activation values
        std::vector< arma::mat > _as;

        // pre-activation/pre-sigmoid values
        std::vector< arma::mat > _zs;

        // error delta values
        std::vector< arma::mat > _ds;

        // theta gradients
        std::vector< arma::mat > _gs;

        // construct the neural network
        void _construct(const NNLayerDesc_t* layerDesc, const unsigned& layerCount);

        // forward propagation
        void _propagate_forward(const arma::mat& X);

        // sigmoid activation function
        arma::mat _sigmoid(const arma::mat& z);

        // cost must be called after forward propagation and before 
        // back propagation in order the have the first delta if training 
        double _network_cost(const arma::mat& y);

        // back propagation
        void _propagate_back(const double& m);

        // sigmoid gradient used in back prop
        arma::mat _sigmoidGradient(const arma::mat& z);

        // generates the display data matrix for X
        arma::mat NNetwork::_get_display_data(const arma::mat X)

    public:
        NNetwork(const NNLayerDesc_t* layerDesc, const unsigned& layerCount);
        virtual ~NNetwork();

        void setLambda(const double lambda);

        double train(const arma::mat& X, const arma::mat& y);

        unsigned predict(const arma::mat& X);

        void visualize();
};
//---------------------------------------------------------
class NNException : public std::runtime_error
{
    private:
        std::string _what;
    public:
        NNException(const char* what) : runtime_error(what), _what(what) {
        }

        virtual const char* what() {
                return _what.c_str();
        }
        
};
//---------------------------------------------------------
} // namespace mlnn


#endif /* NEURALNET_H_ */

