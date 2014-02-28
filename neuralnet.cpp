#include "stdafx.h"
#include "neuralnet.h"


namespace mlnn {

//---------------------------------------------------------
NNetwork::NNetwork(const NNLayerDesc_t* layerDesc, const unsigned& layerCount)  : _lambda(0.0)
{
        _construct(layerDesc, layerCount);
}
//---------------------------------------------------------
NNetwork::~NNetwork()
{
}
//---------------------------------------------------------
void NNetwork::setLambda(const double lambda)
{
    _lambda = lambda;
}
//---------------------------------------------------------
void NNetwork::_construct(const NNLayerDesc_t* layerDesc, const unsigned& layerCount)
{
    if(layerCount < 2) {
        throw NNException("Network must have more than one layer.");
    }
    unsigned i = 1;
    do {
        const NNLayerDesc_t& ilayer = layerDesc[i-1];
        const NNLayerDesc_t& olayer = layerDesc[i];

        _ts.push_back( arma::randu(olayer.unitCount, ilayer.unitCount + 1) );

        i = i + 1;
    } while( i < layerCount );

    _classCount = layerDesc[layerCount - 1].unitCount;
}
//---------------------------------------------------------
void NNetwork::_propagate_forward(const arma::mat& X)
{

    unsigned i = 0;
    std::vector< arma::mat >::const_iterator cii = _ts.begin();
    std::vector< arma::mat >::const_iterator eii = _ts.end();

    _as.push_back( X );

    while(cii != eii){
        const arma::mat& theta = *cii;
        arma::mat z = arma::join_rows(arma::ones(_as[i].n_rows, 1), _as[i]) * theta.t();
        arma::mat a = _sigmoid(z);

        _as.push_back(a);
        _zs.push_back(z);

        cii++;
        i++;
    }
}
//---------------------------------------------------------
arma::mat NNetwork::_sigmoid(const arma::mat& z)
{
    arma::mat nz = -z;
    arma::mat ez = arma::exp(nz);
    arma::mat oz = 1.0 + ez;
    arma::mat rz = 1.0 / oz;

    return rz;
}
//---------------------------------------------------------
double NNetwork::_network_cost(const arma::mat& y)
{
    double m = y.n_rows;
    arma::mat ys = arma::zeros<arma::mat>(y.n_rows, _classCount);
    for(unsigned i = 0; i < y.n_rows; i++){
        for(unsigned j = 0; j < _classCount; j++){
                ys(i,j) = ((y(i) == j) ? 1 : 0);
        }
    }
    // acquire the first pre-delta value with ys
    const arma::mat a = _as.back();

    arma::mat delta = (a - ys);
    _ds.push_back( delta );

    // calculate the regularization parameter
    double rc  = 0.0;

    std::vector<arma::mat>::const_iterator tc = _ts.begin();
    std::vector<arma::mat>::const_iterator te = _ts.end();

    // sum the sums of the squares of the non-biased coefficients
    while(tc != te){
        const arma::mat& theta = *tc;
        const arma::mat ts = theta.cols(1, theta.n_cols - 1);

        rc += arma::accu(arma::square(ts));

        tc++;
    }

    double reg = (_lambda / (2.0 * m)) * rc;

    // std::cout << "Cost reg is " << reg << std::endl;

    arma::mat log_a  = arma::log(a);
    arma::mat neg_ys = -ys;
    arma::mat non_ys = (1.0 - ys);
    arma::mat non_a  = arma::log( (1.0 - a) );

    arma::mat es = ( (neg_ys % log_a) - (non_ys % non_a) ) ;

    // J is the cost or the overall performace of the network
    // this number needs to be as close to 0 as possible
    // for accurate predictions
    double J = (1.0 / m) * arma::accu(es) + reg;

    return J;
}
//---------------------------------------------------------
void NNetwork::_propagate_back(const double& m)
{
    // calculate the delta values of each layer to calculate the gradient need for theta at that layer
    std::vector<arma::mat>::const_reverse_iterator tii = _ts.rbegin();
    std::vector<arma::mat>::const_reverse_iterator tie = _ts.rend();

    unsigned d_index = 0;
    unsigned a_index = _as.size() - 2;
    unsigned z_index = _zs.size() - 2;
    unsigned g_index = _ts.size() - 1;

    _gs.resize(_ts.size());

    while(tii != tie){

        const arma::mat& delta = _ds[d_index];
        const arma::mat a      = arma::join_rows(arma::ones(_as[a_index].n_rows), _as[a_index]);
        const arma::mat& theta = *tii;

        arma::mat reg = arma::join_rows( arma::zeros(theta.n_rows, 1) ,((_lambda / m) * theta.cols(1, theta.n_cols - 1)));
        arma::mat grd = (1.0 / m) * (delta.t() * a);
        _gs[g_index]  = grd + reg;

        if(a_index > 0){
            arma::mat i1 = delta * theta;
            arma::mat i2 = i1.cols(1, i1.n_cols - 1);

            const arma::mat& z     = _zs[z_index];
            const arma::mat next_delta = i2 % _sigmoidGradient(z);

            _ds.push_back(next_delta);

            a_index--;
            d_index++;
            g_index--;
            z_index--;
        }

        tii++;
    }

}
//---------------------------------------------------------
arma::mat NNetwork::_sigmoidGradient(const arma::mat& z)
{
    arma::mat g1 = _sigmoid(z);
    arma::mat g2 = 1.0 - g1;

    return g1 % g2;
}
//---------------------------------------------------------
unsigned NNetwork::predict(const arma::mat& X)
{
    // return the index of the column, e.g. class,  with the heighest prediction/weight
    unsigned r = NNetwork::nprediction;

    double n = 0.0;

    _as.clear();
    _zs.clear();

    _propagate_forward(X);

    const arma::mat& h = _as.back();

    for(unsigned i = 0; i < h.n_cols; i++) {
        if( h(0,i) > n) {
                r = i;
                n = h(0,i);
        } 
    }

    return r;
}
//---------------------------------------------------------
double NNetwork::train(const arma::mat& X, const arma::mat& y)
{
    const double m = X.n_rows;

    // clear out any of the old training data
    _as.clear();
    _ds.clear();
    _gs.clear();
    _zs.clear();

    // forward propagation
    _propagate_forward(X);

    // calulcate the cost and setup the
    // first delta values
    double J = _network_cost(y);

    // propagatge back to get the graidents
    _propagate_back(m);

    for(unsigned i = 0; i < _gs.size(); i++){
        // adjust the theta value with the gradient
        _ts[i] = _ts[i] - _gs[i];
    }

    return J;

}
//---------------------------------------------------------
} // namespace mlnn

