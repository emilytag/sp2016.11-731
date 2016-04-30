#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "cnn/hsm-builder.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

cnn::Dict sourceD;
cnn:: Dict targetD;
int kSOS;
int kSOS2;
int kEOS;
int kEOS2;

unsigned IN_LAYERS = 1;
unsigned OUT_LAYERS = 2;
unsigned INPUT_DIM = 8;  //256
unsigned HIDDEN_DIM = 24;  // 1024
unsigned VOCAB_SIZE = 0;

template <class Builder>
struct EncoderDecoder {
	LookupParameters* p_c;
	Parameters* p_enc2dec;
	Parameters* p_decbias;
	Parameters* p_enc2out;
	Parameters* p_outbias;
	Builder inbuilder;
	Builder outbuilder;

	explicit EncoderDecoder(Model& model): inbuilder(IN_LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
	outbuilder(IN_LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
		p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
		p_enc2dec = model.add_parameters({HIDDEN_DIM * OUT_LAYERS, HIDDEN_DIM});
		p_decbias = model.add_parameters({HIDDEN_DIM * OUT_LAYERS});
		p_enc2out = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
		p_outbias = model.add_parameters({VOCAB_SIZE});
	}
    Expression Encoder(ComputationGraph& cg, const vector<int>& tokens){
    	const unsigned slen = tokens.size();
    	inbuilder.new_graph(cg);
    	inbuilder.start_new_sequence();
    	vector<Expression> embedding;
    	for (unsigned t = 0; t < slen; ++t) {
    		Expression word_lookup = lookup(cg, p_c, tokens[t]);
    		inbuilder.add_input(word_lookup);
    		embedding.push_back(inbuilder.back());
    	}
    	Expression embedding_norm = sum(embedding) / ((float)embedding.size());
    	return embedding_norm;
    }
    void BuildGraph(const vector<int>& source_tokens, const vector<int>& target_tokens, ComputationGraph& cg) {
    	cerr << source_tokens.size() << "\n";
    	const unsigned target_len = target_tokens.size();
    	Expression encoding = Encoder(cg, source_tokens);
    	Expression enc2dec = parameter(cg, p_enc2dec);
    	Expression decbias = parameter(cg, p_decbias);
    	Expression outbias = parameter(cg, p_outbias);
    	Expression enc2out = parameter(cg, p_enc2out);
    	Expression c0 = affine_transform({decbias, enc2dec, encoding});
    	vector<Expression> init(OUT_LAYERS);
    	for (unsigned i = 0; i < OUT_LAYERS; ++i) {
    	      init[i] = pickrange(c0, i * HIDDEN_DIM, i * HIDDEN_DIM + HIDDEN_DIM);
    	      init[i + OUT_LAYERS] = tanh(init[i]);
    	    }
    	outbuilder.new_graph(cg);
    	outbuilder.start_new_sequence(init);
    	vector<Expression> errs(target_tokens.size() + 1);
    	Expression start_token_lookup = lookup(cg, p_c, kSOS2);
    	outbuilder.add_input(start_token_lookup);
    	for (unsigned t = 0; t < target_len; ++t) {
    		Expression outback = outbuilder.back();
    		Expression affine_outback = affine_transform({outbias, enc2out, outback});
    		errs[t] = pickneglogsoftmax(affine_outback, target_tokens[t]);
    		Expression next_token_lookup = lookup(cg, p_c, target_tokens[t]);
    		outbuilder.add_input(next_token_lookup);
    	}
    	Expression outback_last = outbuilder.back();
    	Expression affine_outback_last = affine_transform({outbias, enc2out, outback_last});
    	errs.back() = pickneglogsoftmax(affine_outback_last, kEOS2);
    	sum(errs);
    }
};

void Translate(vector<vector<int>>&  test_sents, Model& model){
  EncoderDecoder<LSTMBuilder> lm(model);
  for(int i = 0; i < test_sents.size(); ++i) {
  	ComputationGraph cg;
  	Expression encoding = lm.Encoder(cg, test_sents[i]);
  	Expression enc2dec = parameter(cg, lm.p_enc2dec);
  	Expression decbias = parameter(cg, lm.p_decbias);
  	Expression outbias = parameter(cg, lm.p_outbias);
  	Expression c0 = affine_transform({decbias, enc2dec, encoding});
  	vector<Expression> init(OUT_LAYERS * 2);
  	for (unsigned i = 0; i < OUT_LAYERS; ++i) {
  	    init[i] = pickrange(c0, i * HIDDEN_DIM, i * HIDDEN_DIM + HIDDEN_DIM);
  	    init[i + OUT_LAYERS] = tanh(init[i]);
  	}
  	lm.outbuilder.new_graph(cg);
  	lm.outbuilder.start_new_sequence(init);
  	Expression start_token_lookup = lookup(cg, lm.p_c, kSOS2);
  	lm.outbuilder.add_input(start_token_lookup);
  	Expression current_word = lm.outbuilder.back();
  	double best = 9e+99;
  	int best_idx = -1;
  	for (int key = 0; key < targetD.size(); ++key){
  		Expression curr_neg_log_softmax = pickneglogsoftmax(current_word, key);
  		double curr_score = as_scalar(cg.incremental_forward());
        if (curr_score < best) {
			best = curr_score;
	 		best_idx = key;
 	    } 

  	}

    while (best_idx != kEOS2) {
        cout << targetD.Convert(best_idx) << " ";
	    Expression token_lookup = lookup(cg, lm.p_c, best_idx);
	    lm.outbuilder.add_input(token_lookup);
	    Expression current_word = lm.outbuilder.back();
	    best = 9e+99; //restart
	    best_idx = -1; //restart
	    for (int key = 0; key < targetD.size(); ++key) {
	      Expression curr_neg_log_softmax = pickneglogsoftmax(current_word, key);
	      double curr_score = as_scalar(cg.incremental_forward());
	      if (curr_score < best) {
	          best = curr_score;
	          best_idx = key;
	      }
	    }

	  }
  } 
}

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);
	bool isTrain = true;
	if (argc == 8){
	  if (!strcmp(argv[1], "-t")) {
	    isTrain = false;
	  }
	}
	if (argc != 5 && argc != 6 && argc != 7 && argc != 8) {
	  cerr << "Usage: " << argv[0] << " train.source train.target dev.source dev.target [test.source] [model.params] [-t]\n";
	  return 1;
	}
	if (isTrain) {
		Model model;
		kSOS = sourceD.Convert("<s>");
		kEOS = sourceD.Convert("</s>");
		kSOS2 = targetD.Convert("<s>");
		kEOS2 = targetD.Convert("</s>");
		vector<vector<int>> train_source, train_target;
		vector<vector<int>> dev_source, dev_target;
		string line;
		{
		  ifstream in(argv[1]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = ReadSentence(line, &sourceD);
		    train_source.push_back(x);
		  }
		}
		sourceD.Freeze();
		{
		  ifstream in(argv[2]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = ReadSentence(line, &targetD);
		    train_target.push_back(x);
		  }
		}
		targetD.Freeze();
		VOCAB_SIZE = targetD.size();
		ofstream out("targetDict");
		boost::archive::text_oarchive oa(out);
		oa << targetD;
		ofstream out2("sourceDict");
		boost::archive::text_oarchive oa2(out2);
		oa2 << sourceD;
		{
		  ifstream in(argv[3]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = ReadSentence(line, &sourceD);
		    dev_source.push_back(x);
		  }
		}
		{
		  ifstream in(argv[4]);
		  assert(in);
		  vector<int> x;
		  string line;
		  while(getline(in, line)) {
		    x = ReadSentence(line, &targetD);
		    dev_target.push_back(x);
		  }
		}	
		Trainer* sgd = new AdamTrainer(&model);
		sgd->eta_decay = 0.08;
		EncoderDecoder<LSTMBuilder> lm(model);
		double best = 9e+99;
		unsigned report_every_i = 100;
	    unsigned dev_every_i_reports = 25;
	    unsigned si = train_source.size();
	    vector<unsigned> order(train_source.size());
	    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
	    bool first = true;
	    int report = 0;
	    double tsize = 0;
	    while(1) {
	          Timer iteration("completed in");
	          double loss = 0;
	          unsigned ttags = 0;
	          double correct = 0;
	          for (unsigned i = 0; i < report_every_i; ++i) {
	            if (si == train_source.size()) {
	              si = 0;
	              if (first) { first = false; } else { sgd->update_epoch(); }
	              cerr << "**SHUFFLE\n";
	              shuffle(order.begin(), order.end(), *rndeng);
	            }
	    		ComputationGraph cg;
	    		const auto& source_sent = train_source[order[si]];
	    		const auto& target_sent = train_target[order[si]];
	    		++si;
	    		ttags += target_sent.size();
	    		tsize += 1;
	    		lm.BuildGraph(source_sent, target_sent, cg);
	    		loss += as_scalar(cg.incremental_forward());
				cg.backward();
				sgd->update(1.0);
			}
			sgd->status();
			cerr << " E = " << (loss / ttags) << " ppl=" << exp(loss / ttags) << " (acc=" << (correct / ttags) << ") ";
			report++;
			if (report % dev_every_i_reports == 0) {
			       double dloss = 0;
			       unsigned dtags = 0;
			       double dcorr = 0;
			       for (unsigned di = 0; di < dev_source.size(); ++di) {
			         const auto& x = dev_source[di];
			         const auto& y = dev_target[di];
			         ComputationGraph cg;
			         lm.BuildGraph(x, y, cg);
			         dloss += as_scalar(cg.incremental_forward());
			         dtags += y.size();
			       }
			       if (dloss < best) {
			         cerr << endl;
			         cerr << "Current dev performance exceeds previous best, saving model" << endl;
			         best = dloss;
			         ofstream out("EDmodel");
			         boost::archive::text_oarchive oa(out);
			         oa << model;
			       }
			       cerr << "\n***DEV [epoch=" << (tsize / (double)train_source.size()) << "] E = " << (dloss / dtags) << " ppl=" << exp(dloss / dtags) << " acc=" << (dcorr / dtags) << ' ';
			     }
			 }
		}
	else {
	 Model model;
	 ifstream in("sourceDict");
	 assert(in);
	 boost::archive::text_iarchive ia(in);
	 ia >> sourceD;

	 ifstream in3("targetDict");
	 assert(in3);
	 boost::archive::text_iarchive ia3(in3);
	 ia3 >> targetD;

	 sourceD.Freeze();
	 targetD.Freeze();
	 VOCAB_SIZE = targetD.size();

	 vector<vector<int>> test_source;

	 ifstream in2("EDmodel");
	 assert(in2);
	 boost::archive::text_iarchive ia2(in2);
	 ia2 >> model;
	 {
	   ifstream in(argv[5]);
	   assert(in);
	   vector<int> x;
	   string line;
	   while(getline(in, line)) {
	     x = ReadSentence(line, &sourceD);
	     test_source.push_back(x);
	   }
	 }
	 Translate(test_source, model);
	 }
}
//train on the train+dev data duh

