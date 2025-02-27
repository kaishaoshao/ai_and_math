/***
*                                                              
*                                                              
*       ,-.                           ,-.                      
*   ,--/ /|              ,--,     ,--/ /|              ,--,    
* ,--. :/ |            ,--.'|   ,--. :/ |            ,--.'|    
* :  : ' /             |  |,    :  : ' /             |  |,     
* |  '  /    ,--.--.   `--'_    |  '  /    ,--.--.   `--'_     
* '  |  :   /       \  ,' ,'|   '  |  :   /       \  ,' ,'|    
* |  |   \ .--.  .-. | '  | |   |  |   \ .--.  .-. | '  | |    
* '  : |. \ \__\/: . . |  | :   '  : |. \ \__\/: . . |  | :    
* |  | ' \ \," .--.; | '  : |__ |  | ' \ \," .--.; | '  : |__  
* '  : |--'/  /  ,.  | |  | '.'|'  : |--'/  /  ,.  | |  | '.'| 
* ;  |,'  ;  :   .'   \;  :    ;;  |,'  ;  :   .'   \;  :    ; 
* '--'    |  ,     .-./|  ,   / '--'    |  ,     .-./|  ,   /  
*          `--`---'     ---`-'           `--`---'     ---`-'   
*                                                              
*/

#include <gtest/gtest.h>  // 引入Google Test框架的头文件
#include <glog/logging.h> // 引入Google glog库的头文件

int main(int argc, char *argv[]) {
  // 初始化Google Test框架
  testing::InitGoogleTest(&argc, argv);
  // 初始化Google glog库
  google::InitGoogleLogging("Kuiper");
  // 设置日志文件输出目录
  FLAGS_log_dir = "../1.course/log";
  // 设置日志同时输出到标准错误
  FLAGS_alsologtostderr = true;

  // 输出日志信息，表示测试开始
  LOG(INFO) << "Start test...\n";
  // 运行所有测试用例
  return RUN_ALL_TESTS();
}