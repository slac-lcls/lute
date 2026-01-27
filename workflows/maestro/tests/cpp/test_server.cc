#include "server.hh"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <chrono>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>

using namespace HTTP;

class MockHandler : public Handler {
public:
  Response operator()(const Request& request) override {
    Response res(CODE::OK);
    res.set_content("Echo: " + request.content());
    return res;
  }
};

TEST(ServerTest, StartStop) {
  // Use a high port to avoid permission issues
  Server server("127.0.0.1", 18888);

  server.start();
  EXPECT_TRUE(server.running());
  EXPECT_EQ(server.port(), 18888);
  EXPECT_EQ(server.host(), "127.0.0.1");

  server.stop();
  EXPECT_FALSE(server.running());
}

TEST(ServerTest, HandleRequest) {
  Server server("127.0.0.1", 18889);
  server.add_request_handler("/echo",
                             METHOD::POST,
                             std::make_shared<MockHandler>());
  server.start();

  // Small sleep to ensure server is up
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Simple client to test connection
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(18889);
  inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

  ASSERT_GE(connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)), 0);

  std::string request = "POST /echo HTTP/1.1\r\nContent-Length: 5\r\n\r\nHello";
  send(sock, request.c_str(), request.length(), 0);

  char buffer[1024] = {0};
  read(sock, buffer, 1024);
  std::string response(buffer);

  EXPECT_NE(response.find("200 OK"), std::string::npos);
  EXPECT_NE(response.find("Echo: Hello"), std::string::npos);

  close(sock);
  server.stop();
}
