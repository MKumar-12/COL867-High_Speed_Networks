package main

import (
	"fmt"
	"net"
	"time"
	"syscall"
	"os"
	"encoding/csv"
)


// set priority of sender process(pid = 0) to -10 		{increase its priority, to avoid Skipping of packets}
func setProcessPriority() error {
	err := syscall.Setpriority(syscall.PRIO_PROCESS, 0, -10)
	if err != nil {
		return fmt.Errorf("Failed to set priority:", err)
	} 	
	fmt.Println("Sender priority set to -10")
	return nil
}

// initialize UDP socket 
func setupUDPReceiver() (*net.UDPConn, error) {
	// Create UDP socket -> bind to the receiver IP and UDP port 
	udpAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", ReceiverIP, UDPPort))
	if err != nil {
		return nil, fmt.Errorf("Error resolving UDP address:", err)
	}

	// Listen for incoming UDP packets
	udpConn, err := net.DialUDP("udp", nil, udpAddr)
	if err != nil {
		return nil, fmt.Errorf("Error creating UDP socket:", err)
	}

	// Set the UDP socket's receiver buffer size to 1 MB
	udpConn.SetReadBuffer(1024 * 1024)
	
	fmt.Println("UDP socket is running on port %d...", UDPPort)
	return udpConn, nil
}

// set up a TCP connection for receiving delta_r
func setupTCPListener() (*net.TCPConn, error) {
	tcpAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("%s:%d", ReceiverIP, TCPPort))
	if err != nil {
		return nil, fmt.Errorf("error resolving TCP address: %v", err)
	}

	// Connect to the TCP receiver
	tcpConn, err := net.DialTCP("tcp", nil, tcpAddr)
	if err != nil {
		return nil, fmt.Errorf("error starting TCP server: %v", err)
	}

	tcpConn.SetNoDelay(true)            		// Prevent message coalescing
	tcpConn.SetWriteBuffer(1024 * 1024) 		// Increase TCP send buffer
	tcpConn.SetReadBuffer(1024 * 1024)

	fmt.Println("TCP server is running. Listening on port %d...", TCPPort)
	return tcpConn, nil
}


func runSender() {
	// Set process priority to -10
	if err := setProcessPriority(); err != nil {
		fmt.Println(err)
		return
	}

	// setup UDP receiver
	udpConn, err := setupUDPReceiver()
	if err != nil {
		fmt.Println("Error setting up UDP receiver:", err)
		return
	}
	defer udpConn.Close()

	// Set up TCP listener
	tcpConn, err := setupTCPListener()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer tcpConn.Close()

	// Open CSV file in append mode
	file, err := os.OpenFile(DATA_DIR+"/stream_rates_rin.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println("Error opening stream_rates_rin CSV file:", err)
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()


	// Collect Stats
	var achievedRates []string

	// Send UDP probing packet
	fmt.Println("Sending UDP probing packets...")
	startTime := time.Now()


	for i := 0; i < NumPackets; i++ {
		_, err := udpConn.Write(make([]byte, PacketSize))
		if err != nil {
			fmt.Println("Error sending packet:", err)
		}
		// time.Sleep(time.Nanosecond) // Sending delay
	}

	endTime := time.Now()

	// Compute r_in
	duration := endTime.Sub(startTime).Seconds()
	fmt.Println("Start time : ", startTime)
	fmt.Println("End time   : ", endTime)
	fmt.Println("Duration   = ", duration)
	fmt.Println("Load Sent  = ", PacketSize*NumPackets*8)
	
	rIn := float64(PacketSize*NumPackets*8) / (duration * 1_000_000) 		// Mbps conversion
	fmt.Printf("Estimated r_in: %.2f mbps\n", rIn)

	achievedRates = append(achievedRates, fmt.Sprintf("%.2f", rIn))			// update received rates with computed value of r_in
	
	// Receive delta_r over TCP
	buffer := make([]byte, 1024)
	n, err := tcpConn.Read(buffer)
	if err != nil {
		fmt.Println("Error receiving delta_r:", err)
		return
	}

	var deltaR float64
	fmt.Sscanf(string(buffer[:n]), "%f", &deltaR)
	fmt.Printf("Received delta_r: %.2f mbps\n", deltaR)

	// Sending 20 streams sequentially of rate k * delta_r, and verify speed
	fmt.Println("Sending 20 streams of varying rates...")
	for k := 1; k <= 20; k++ {
		streamRate := float64(k) * deltaR                   // Rate in Mbps
		inter := float64(PacketSize*8) / (streamRate * 1e6) // Interval in seconds
		intervalNs := time.Duration(inter * 1e9)            // Convert to nanoseconds

		fmt.Printf("Stream %d: Sending at %.2f Mbps (interval: %v)\n", k, streamRate, intervalNs)

		// Send TCP message to indicate stream start
		_, err := tcpConn.Write([]byte(fmt.Sprintf("Stream %d starting", k)))
		if err != nil {
			fmt.Printf("Error sending TCP message for stream %d: %v\n", k, err)
		}

		// Send UDP packets (pkt train, with n = 100)
		packetSentTimes := []time.Time{}
		streamStart := time.Now()
		sentBytes := 0

		for i := 0; i < NumPackets; i++ {
			_, err := udpConn.Write(make([]byte, PacketSize))
			if err != nil {
				fmt.Printf("Error sending packet in stream %d: %v\n", k, err)
			}
			
			sentBytes += PacketSize * 8
			packetSentTimes = append(packetSentTimes, time.Now())
			
			nextSendTime := streamStart.Add(time.Duration(i+1) * intervalNs)
			// Wait until the next send time
			for time.Since(nextSendTime) < 0 {
				// Empty loop for precise wait
			}
		}

		streamEnd := time.Now()
		streamDuration := streamEnd.Sub(streamStart).Seconds()
		expectedDuration := float64(NumPackets) * inter

		fmt.Printf("Stream %d duration: %.6f sec (Expected: %.6f sec)\n", k, streamDuration, expectedDuration)

		// Compute r_in for this stream
		totalInterArrival := 0.0
		if len(packetSentTimes) > 1 {
			for i := 1; i < len(packetSentTimes); i++ {
				totalInterArrival += packetSentTimes[i].Sub(packetSentTimes[i-1]).Seconds()
			}
			
			avgInterArrival := totalInterArrival / float64(len(packetSentTimes)-1)
			streamRin := (float64(PacketSize*8) / (avgInterArrival * 1e6))
			
			fmt.Printf("Stream %d: Achieved r_in: %.2f Mbps\n", k, streamRin)
			achievedRates = append(achievedRates, fmt.Sprintf("%.2f", streamRin))
		} else {
			fmt.Printf("Stream %d: Not enough packets to compute r_out\n", k)
			achievedRates = append(achievedRates, "0.00")
		}

		time.Sleep(1 * time.Second)
	}

	// Write the achieved rates to the CSV file
	writer.Write(achievedRates)
	writer.Flush()
	fmt.Println("Achieved rates written to CSV!")

	// Wait to ensure all streams finish before exiting
	time.Sleep(3 * time.Second)
}