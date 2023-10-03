from model.magnn_model import train_magnn_embedding
from model.price_lstm_embedding import train_price_embedding
from prepossessing.data_prepare import init_event_news_embeeding_df,get_magnn_result


from prepossessing.constant import TEST_QUARTERS
import time



def main():    
    
    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))   
    
    
    for i in range(1):
        test_period_tuple = TEST_QUARTERS[0]
        train_period_tuples = TEST_QUARTERS[0]
        
        """data preparation"""
        # we use lstm to do price embeddings here
        train_price_embedding(TEST_QUARTERS[0], TEST_QUARTERS[0])
    
        # then we do event&news embeddings
        init_event_news_embeeding_df(TEST_QUARTERS[0][0], TEST_QUARTERS[0][1], 'event')
        init_event_news_embeeding_df(TEST_QUARTERS[0][0], TEST_QUARTERS[0][1], 'news')
    
        """model traning"""
        # do magnn training here
        train_magnn_embedding(train_period_tuples, test_period_tuple, timestamp)
        
        """reselts"""
        df=get_magnn_result(TEST_QUARTERS[0])
        df.to_csv('magnn_result.csv')
        
        print("magnn result has been saved to \'magnn_result.csv\' ")

if __name__=='__main__':
    main()
        