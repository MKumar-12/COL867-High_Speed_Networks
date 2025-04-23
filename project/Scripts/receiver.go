package main

import (
	"fmt"
	"net"
	"time"
	"syscall"
	"os"
	"encoding/csv"
)


// set priority of receiving process(pid = 0) to -5 		{increase its priority, to avoid Skipping of packets}
func setProcessPriority_R() error {
	err := syscall.Setpriority(syscall.PRIO_PROCESS, 0, -5)
	if err != nil {
		return fmt.Errorf("Failed to set priority:", err)
	} 	
	fmt.Println("Receiver priority set to -5")
	return nil
}

// initialize UDP receiver socket with required settings
func setupUDPReceiver_R() (*net.UDPConn, error) {
	// Create UDP socket -> bind to the receiver IP and UDP port 
	udpAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", ListenIP, UDPPort))
	if err != nil {
		return nil, fmt.Errorf("Error resolving UDP address:", err)
	}

	// Listen for incoming UDP packets
	udpConn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return nil, fmt.Errorf("Error creating UDP socket:", err)
	}
	
	// Set the UDP socket's receiver buffer size to 1 MB
	// udpConn.SetReadBuffer(1024 * 1024)

	// Get the file descriptor for the socket
	fd, err := udpConn.File()
	if err != nil {
		udpConn.Close()
		return nil, fmt.Errorf("Error getting file descriptor:", err)
	}
	defer fd.Close()

	// Reduce OS receive buffer to 512KB manually (default 8KB Windows, 208KB LINUX) -> improve latency for small pkt streams
	err = syscall.SetsockoptInt(int(fd.Fd()), syscall.SOL_SOCKET, syscall.SO_RCVBUF, 512*1024)
	if err != nil {
		udpConn.Close()
		return nil, fmt.Errorf("Error setting SO_RCVBUF:", err)
	}

	// Enable SO_REUSEPORT to allow multiple sockets(processes/threads) to bind to the same UDP port
	const SO_REUSEPORT = 15
	err = syscall.SetsockoptInt(int(fd.Fd()), syscall.SOL_SOCKET, SO_REUSEPORT, 1)
	if err != nil {
		udpConn.Close()
		return nil, fmt.Errorf("Error setting SO_REUSEPORT:", err)
	}

	// Set timeout for reading pkt from the UDP socket -> if no pkt received in 100ms, return timeout error -> prevent blocking
	// udpConn.SetReadDeadline(time.Now().Add(100 * time.Millisecond))

	fmt.Println("UDP receiver is running on port %d...", UDPPort)
	return udpConn, nil
}

// set up a TCP listener
func setupTCPListener_R() (*net.TCPListener, error) {
	tcpAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("%s:%d", ListenIP, TCPPort))
	if err != nil {
		return nil, fmt.Errorf("error resolving TCP address: %v", err)
	}

	tcpListener, err := net.ListenTCP("tcp", tcpAddr)
	if err != nil {
		return nil, fmt.Errorf("error starting TCP server: %v", err)
	}

	fmt.Println("TCP server is running. Waiting for sender to connect...")
	return tcpListener, nil
}

// wait for an incoming TCP connection
func waitForTCPConnection(listener *net.TCPListener) (net.Conn, error) {
	tcpConn, err := listener.Accept()
	if err != nil {
		return nil, fmt.Errorf("error accepting TCP connection: %v", err)
	}

	fmt.Println("Received conn. from: ", tcpConn.RemoteAddr())
	fmt.Println("TCP connection established. Now waiting for UDP probing packets...")
	return tcpConn, nil
}


func runReceiver() {
	// Set process priority to -5
	if err := setProcessPriority_R(); err != nil {
		fmt.Println(err)
		return
	}
	
	// setup UDP receiver
	udpConn, err := setupUDPReceiver_R()
	if err != nil {
		fmt.Println("Error setting up UDP receiver:", err)
		return
	}
	defer udpConn.Close()

	// Set up TCP listener
	tcpListener, err := setupTCPListener_R()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer tcpListener.Close()

	// Wait for a TCP connection before proceeding
	tcpConn, err := waitForTCPConnection(tcpListener)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer tcpConn.Close()

	// Open CSV file for writing
	file, err := os.OpenFile(DATA_DIR+"/stream_rates_rout.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println("Error opening stream_rates_rout CSV file:", err)
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()


	// Collect Stats (for incoming UDP packets)
	var receivedRates []string
	buffer := make([]byte, BufferSize)
	receivedBytes := 0
	packetCount := 0
	
	startTime := time.Now()
	lastPacketTime := time.Now()
	arrivalTimes := []time.Time{} 
	var totalInterArrival float64
	var interArrivalTimes []float64

	for {
		// udpConn.SetReadDeadline(time.Now().Add(Timeout))
		n, _, err := udpConn.ReadFromUDP(buffer)
		if err != nil {
			if netErr, ok := err.(*net.OpError); ok && netErr.Timeout() {
				fmt.Println("Timeout reached! Stopped receiving UDP Packets")
			} else {
				fmt.Println("Error receiving UDP packet:", err)
			}
			break
		}

		receivedBytes += n
		packetCount++
		// fmt.Printf("Received packet %d, Total bytes received: %d\n", packetCount, n)
		
		currentTime := time.Now()
		arrivalTimes = append(arrivalTimes, currentTime)
		
		if packetCount > 1 { 	// Skip first packet as there's no previous timestamp
        	interArrivalTime := currentTime.Sub(lastPacketTime).Seconds()
        	interArrivalTimes = append(interArrivalTimes, interArrivalTime)
        	totalInterArrival += interArrivalTime
    	}

		lastPacketTime = currentTime		// Update last received packet time

		// stop receiving packets after NumPackets
		if packetCount >= NumPackets {
			fmt.Println("All burst packets received.")
			break
		}
	}


	// Received stats
	fmt.Println("start time = ", startTime)
	endTime := lastPacketTime
	fmt.Println("end time   = ", endTime)

	duration := endTime.Sub(startTime).Seconds()
	fmt.Println("duration   = ", duration)
	fmt.Println("load received in bits = ", receivedBytes*8)

	// Compute r_out
	rOut := 0.0
	if duration > 0 {
		rOut = (float64(receivedBytes) * 8) / (duration * 1_000_000) // Mbps conversion
	}
	fmt.Printf("Estimated r_out: %.2f Mbps\n", rOut)

	// Compute r_out using inter-arrival times
	// if len(interArrivalTimes) > 0 {
	// 	avgInterArrival := totalInterArrival / float64(len(interArrivalTimes))
	// 	rOutIAT := (float64(receivedBytes*8) / (avgInterArrival * 1e6)) // Mbps conversion
	// 	fmt.Printf("Estimated r_out (using IAT): %.2f Mbps\n", rOutIAT)
	// } else {
	// 	fmt.Println("Not enough packets to compute r_out using IAT.")
	// }

	// Compute delta_r
	deltaR := rOut / DeltaRK
	if deltaR == 0 {
		deltaR = 1.0 		// Default to 1 Mbps if no packets received
	}
	fmt.Printf("Computed delta_r: %.2f Mbps\n", deltaR)

	receivedRates = append(receivedRates, fmt.Sprintf("%.2f", rOut))			// update received rates with computed value of r_out

	// Send computed delta_r to SENDER over TCP
	_, err = fmt.Fprintf(tcpConn, "%.2f", deltaR)
	if err != nil {
		fmt.Println("Error sending delta_r:", err)
	}

	// Receiving 20 streams sequentially of rate k * delta_r
	fmt.Println("Waiting for 20 streams...")
	for k := 1; k <= DeltaRK; k++ {
		buffer := make([]byte, BufferSize)
		
		// Wait for TCP start signal indicating beginning of stream
		_, err := tcpConn.Read(buffer)
		if err != nil {
			fmt.Printf("Error reading stream %d start signal: %v\n", k, err)
			return
		}
		fmt.Printf("Stream %d start signal received. Now receiving UDP packets...\n", k)

		// Compute stream rate
		streamBytes := 0
		streamPackets := 0
		streamArrivalTimes := []time.Time{}
		
		for {
			timestart := time.Now()

			n, _, err := udpConn.ReadFromUDP(buffer)
			if err != nil {
				fmt.Println("Time taken for timeout = ", time.Now().Sub(timestart).Seconds())
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					fmt.Printf("Stream %d: Timeout reached! Stopping reception.\n", k)
				} else {
					fmt.Printf("Stream %d: Error receiving UDP packets: %v\n", k, err)
				}
				break
			}

			streamBytes += n
			streamPackets++
			streamArrivalTimes = append(streamArrivalTimes, time.Now())

			// Stop receiving packets after expected packets are received
			if streamPackets >= NumPackets {
				fmt.Println("All stream packets received.")
				break
			}
		}

		// Compute r_out for this stream
		totalInterArrival = 0
		for i := 1; i < len(streamArrivalTimes); i++ {
			totalInterArrival += streamArrivalTimes[i].Sub(streamArrivalTimes[i-1]).Seconds()
		}

		if len(streamArrivalTimes) > 1 {
			avgInterArrival := totalInterArrival / float64(len(streamArrivalTimes)-1)
			streamRout := (float64(PacketSize*8) / (avgInterArrival * 1e6))
			
			fmt.Printf("Stream %d: Estimated r_out using inter-arrival time: %.2f Mbps\n", k, streamRout)
			receivedRates = append(receivedRates, fmt.Sprintf("%.2f", streamRout))
		} else {
			fmt.Printf("Stream %d: Not enough packets to compute r_out\n", k)
		}
	}

	// Write the received rates to the CSV file
	writer.Write(receivedRates)
	fmt.Println("Achieved rates written to CSV!")
}