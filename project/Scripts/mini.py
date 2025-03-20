import sys
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import time, os
import csv

class SimpleLinkTopo(Topo):
    def build(self, availbw):
        # Add two hosts
        h1 = self.addHost('h1')  # Server
        h2 = self.addHost('h2')  # Client

        h3 = self.addHost('h3')  # Server
        h4 = self.addHost('h4')  # Client
        # Add a single switch
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')


        # Add links with 100 Mbps bandwidth, no loss, and no delay
        self.addLink(h1, s1, bw=availbw, loss=0, delay='0ms')
        self.addLink(h2, s2, bw=availbw, loss=0, delay='0ms')
        self.addLink(h3, s1, bw=availbw, loss=0, delay='0ms')
        self.addLink(h4, s2, bw=availbw, loss=0, delay='0ms')
        self.addLink(s1, s2, bw=availbw, loss=0, delay='0ms')



def run(availbw, crosstraffic):
    # Set the log level to info to see detailed output
    setLogLevel('info')
    
    # IP and port of the remote controller
    controller_ip = '10.17.5.63'  # Change to the controller's IP address if not local
    controller_port = 6653       # Default OpenFlow controller port
    
    # Initialize the topology
    topo = SimpleLinkTopo(availbw=availbw)

    # Initialize the network with the custom topology and TCLink for link configuration
    net = Mininet(topo=topo, link=TCLink, controller=None)

    # Add the remote controller to the network
    remote_controller = RemoteController('c0', ip=controller_ip, port=controller_port)
    net.addController(remote_controller)

    # Start the network
    net.start()

    # Get references to h1 and h2
    h1 = net.get('h1')
    h2 = net.get('h2')

    print("IP address of h1:", h1.IP())
    print("IP address of h2:", h2.IP())
    
    h3 = net.get('h3')
    h4 = net.get('h4')

    print("IP address of h3:", h3.IP())
    print("IP address of h4:", h4.IP())


    # Start the receiver (client) on h2
    SERVER_IP = "10.0.0.1"  # Can be adjusted as needed
    SERVER_PORT = 6555

    print(f"\n-------------introducing traffic---------\n")
    # CLI(net)
    h4.cmd(f"iperf -s -u > h4_iperf.txt 2>&1 &")
    time.sleep(2)
    h3.cmd(f"iperf -c {h4.IP()} -u -b {crosstraffic}M -t 29 > h3_iperf.txt 2>&1 &")   # CLI(net)
    h1.cmd(f"export PATH=$PATH:/usr/local/go/bin")
    h2.cmd(f"export PATH=$PATH:/usr/local/go/bin")

    # CLI(net)
    print("--- Starting receiver (client) on h2 ---")
    h2.cmd(f"go run temp.go > h2_output.txt 2>&1 &")

    # h2.cmd(f"iperf -s > h2_output.txt 2>&1 &")
    # time.sleep(2)
    # h1.cmd(f"iperf -c {h2.IP()} -t 12 > h1_output.txt 2>&1 &")
    #  CLI(net) 
    # Wait for a short period to ensure receiver has started first
    time.sleep(1)

    # Start the sender (server) on h1
    print("--- Starting sender (server) on h1 ---")
    h1.cmd(f"go run sender.go> h1_output.txt 2>&1 &")

    # print(f"\n-------------introducing traffic---------\n")
    # h1.cmd(f"iperf -s > h1_iperf.txt &")
    # h2.cmd(f"iperf -c {h1.IP()} -b 20M -t 20 > h2_iperf.txt &")
    # Wait for the server to finish

    time.sleep(30) 
    # Stop the network
    net.stop()


    data = [availbw, crosstraffic]  # This will be a row in CSV
    # Open the CSV file in append mode
    with open("test_info.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


    print("--- Test completed, receiver and sender programs finished ---")
    
    # Optionally, you can run Mininet's CLI for manual interaction if needed
    # CLI(net)

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage: sudo python sinple_topo.py <bandwidth> <crosstraffic>")
        sys.exit(1)

    bandwidth=float(sys.argv[1])
    crosstraffic= float(sys.argv[2])
    run(availbw= bandwidth, crosstraffic= crosstraffic)
