g++ your_team_name/gcn_v1.cpp -o your_team_name/gcn_v1
./your_team_name/gcn_v1 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin
g++ your_team_name/gcn.cpp -o your_team_name/gcn
./your_team_name/gcn 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin