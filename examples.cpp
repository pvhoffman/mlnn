#include "stdafx.h"
#include "neuralnet.h"

// create a neural network to predict the result
// of an AND gate with two binary inputs and 1 binary output
void ANDNetworkExample()
{
    // 1 input layer + 1 output layer
    // input layer with two units as the two operands of the AND binary op
    // output layer with two units as predicting either a 0 or 1
    mlnn::NNetwork::NNLayerDesc_t AndDesc[] = { {2}, {2} };
    mlnn::NNetwork AndNetwork(AndDesc, 2);
    AndNetwork.setLambda(0.0);


    double J = 0;

    // create some data with which to train
    arma::mat X = arma::zeros(5000, 2);
    arma::mat y = arma::zeros(5000, 1);

    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a & b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    // 1000 iterations might do
    for(unsigned i = 0; i< 1000; i++){
        J = AndNetwork.train(X,y);
        if( !(i % 100) ){
            std::cout << "Current cost for AND network at training iteration " << i << " is " << J << std::endl;
        }
    }

    std::cout << "Final cost for AND network at end training is " << J << std::endl;

    // create new data to test
    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a & b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    unsigned badp = 0;

    // check the predictions
    for( unsigned i = 0; i < 5000; i++){
        arma::mat px = X.row(i);

        unsigned p = AndNetwork.predict(px);
        unsigned a = (unsigned)X(i,0) & (unsigned)X(i,1);

        if(p != a){
                std::cout << "Incorrect prediction at " << i << ". P is " << p << " and A is " << a << "." << std::endl;
                badp++;
        }

    }
    
    std::cout << badp << " incorrect predictions from trained AND network." << std::endl << std::endl;
}

// create a neural network to predict the result
// of an OR gate with two binary inputs and 1 binary output
void ORNetworkExample()
{
    // 1 input layer + 1 output layer
    // input layer with two units as the two operands of the OR binary op
    // output layer with two units as predicting either a 0 or 1
    mlnn::NNetwork::NNLayerDesc_t OrDesc[] = { {2}, {2} };
    mlnn::NNetwork OrNetwork(OrDesc, 2);
    OrNetwork.setLambda(0.0);


    double J = 0;

    // create some data with which to train
    arma::mat X = arma::zeros(5000, 2);
    arma::mat y = arma::zeros(5000, 1);

    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a | b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    // 1000 iterations might do
    for(unsigned i = 0; i< 1000; i++){
        J = OrNetwork.train(X,y);
        if( !(i % 100) ){
            std::cout << "Current cost for OR network at training iteration " << i << " is " << J << std::endl;
        }
    }

    std::cout << "Final cost for OR network at end training is " << J << std::endl;

    // create new data to test
    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a | b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    unsigned badp = 0;

    // check the predictions
    for( unsigned i = 0; i < 5000; i++){
        arma::mat px = X.row(i);

        unsigned p = OrNetwork.predict(px);
        unsigned a = (unsigned)X(i,0) | (unsigned)X(i,1);

        if(p != a){
                std::cout << "Incorrect prediction at " << i << ". P is " << p << " and A is " << a << "." << std::endl;
                badp++;
        }

    }
    
    std::cout << badp << " incorrect predictions from trained OR network." << std::endl << std::endl;
}

// create a neural network to predict the result
// of an XOR gate with two binary inputs and 1 binary output
void XORNetworkExample()
{
    // XOR requires more complexity hence a NN with a hidden layer
    // 1 input layer + 1 hidden layer + 1 output layer
    // input layer with two units as the two operands of the OR binary op
    // hidden layer for added coeffecients et al to match XOR complexity
    // output layer with two units as predicting either a 0 or 1
    mlnn::NNetwork::NNLayerDesc_t XorDesc[] = { {2}, {2}, {2} };
    mlnn::NNetwork XorNetwork(XorDesc, 3);
    XorNetwork.setLambda(0.0);


    double J = 0;

    // create some data with which to train
    arma::mat X = arma::zeros(5000, 2);
    arma::mat y = arma::zeros(5000, 1);

    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a ^ b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    // 1000 iterations might do
    for(unsigned i = 0; i< 1000; i++){
        J = XorNetwork.train(X,y);
        if( !(i % 100) ){
            std::cout << "Current cost for XOR network at training iteration " << i << " is " << J << std::endl;
        }
    }

    std::cout << "Final cost for XOR network at end training is " << J << std::endl;

    // create new data to test
    for( unsigned i = 0; i < 5000; i++){
        unsigned a = ( rand() & 1 ) ? 1 : 0;
        unsigned b = ( rand() & 1 ) ? 1 : 0;
        unsigned c = ( a ^ b ) ? 1 : 0;

        X(i, 0) = a;
        X(i, 1) = b;
        y(i)    = c;
    }

    unsigned badp = 0;

    // check the predictions
    for( unsigned i = 0; i < 5000; i++){
        arma::mat px = X.row(i);

        unsigned p = XorNetwork.predict(px);
        unsigned a = (unsigned)X(i,0) ^ (unsigned)X(i,1);

        if(p != a){
                std::cout << "Incorrect prediction at " << i << ". P is " << p << " and A is " << a << "." << std::endl;
                badp++;
        }

    }
    
    std::cout << badp << " incorrect predictions from trained XOR network." << std::endl << std::endl;
}



int main (int argc, char const* argv[])
{
    ANDNetworkExample();
    ORNetworkExample();
    XORNetworkExample();
    return 0;
}
