package main

import (
	"fmt"
	"net"
	"time"
	"syscall"
	"os"
	"encoding/csv"
	// "golang.org/x/sys/unix"
)

const (
	ReceiverIP = "10.0.0.2"
	ListenIP   = "10.0.0.2"
	//ReceiverIP = "10.17.5.63"
	//ListenIP   = "10.17.5.63"
	UDPPort    = 5005
	TCPPort    = 6000
	PacketSize = 1500
	NumPackets = 100
	DeltaRK    = 20
	BufferSize = 2048
	Timeout    = 2 * time.Second
)

// func setRealtimePriority() {
// 	param := unix.SchedParam{Priority: 99} // Highest real-time priority
// 	err := unix.SchedSetscheduler(0, unix.SCHED_FIFO, &param)
// 	if err != nil {
// 		fmt.Println("Failed to set real-time priority:", err)
// 	}
// }

func main() {
	// Create UDP socket
	// setRealtimePriority()
	err := syscall.Setpriority(syscall.PRIO_PROCESS, 0, -10)
	if err != nil {
		fmt.Println("Failed to set priority:", err)
	} else {
		fmt.Println("Sender priority set to -10")
	}
	udpAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", ReceiverIP, UDPPort))
	if err != nil {
		fmt.Println("Error resolving UDP address:", err)
		return
	}

	udpConn, err := net.DialUDP("udp", nil, udpAddr)
	if err != nil {
		fmt.Println("Error creating UDP socket:", err)
		return
	}
	defer udpConn.Close()
	udpConn.SetReadBuffer(1024 * 1024)

	// Create TCP connection for receiving delta_r
	tcpAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("%s:%d", ReceiverIP, TCPPort))
	if err != nil {
		fmt.Println("Error resolving TCP address:", err)
		return
	}

	tcpConn, err := net.DialTCP("tcp", nil, tcpAddr)
	if err != nil {
		fmt.Println("Error connecting to TCP socket:", err)
		return
	}
	defer tcpConn.Close()

	tcpConn.SetNoDelay(true)            // Prevent message coalescing
	tcpConn.SetWriteBuffer(1024 * 1024) // Increase TCP send buffer
	tcpConn.SetReadBuffer(1024 * 1024)

	// Open CSV file in append mode
	file, err := os.OpenFile("stream_rates_rin.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println("Error opening stream_rates_rin CSV file:", err)
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

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
	fmt.Println("start time : ", startTime)
	fmt.Println("end time : ", endTime)
	fmt.Println("duration = ", duration)
	fmt.Println("load sent = ", PacketSize*NumPackets*8)
	rIn := float64(PacketSize*NumPackets*8) / (duration * 1e6)
	fmt.Printf("Estimated r_in: %.2f mbps\n", rIn)

	achievedRates = append(achievedRates, fmt.Sprintf("%.2f", rIn))
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

	// Send 20 streams of rate k * delta_r and verify speed
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

		// time.Sleep(1 * time.Millisecond)

		// buffer := make([]byte, 1024)
		// tcpConn.SetReadDeadline(time.Now().Add(2 * time.Second)) // Timeout if no ACK received
		// n, err := tcpConn.Read(buffer)
		// if err != nil {
		// 	fmt.Println("Warning: ACK not received, potential buffer issue")
		// } else {
		// 	fmt.Printf("ACK received: %s\n", string(buffer[:n]))
		// }

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
			// Busy-wait loop to maintain precise timing
			nextSendTime := streamStart.Add(time.Duration(i+1) * intervalNs)
			// for time.Now().Before(nextSendTime) {
			// 	// Actively wait without sleeping
			// }
			// for time.Until(nextSendTime) > 0 {
			// 	runtime.Gosched() // Yield the processor to improve timing accuracy
			// }

			for time.Since(nextSendTime) < 0 {
				// Empty loop for precise wait
			}
		}

		streamEnd := time.Now()
		streamDuration := streamEnd.Sub(streamStart).Seconds()
		expectedDuration := float64(NumPackets) * inter

		fmt.Printf("Stream %d duration: %.6f sec (Expected: %.6f sec)\n", k, streamDuration, expectedDuration)

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

		// fmt.Printf("Stream %d sent bytes: %d\n", k, sentBytes)

		//achievedRate := float64(sentBytes) / (streamDuration * 1e6)
		//fmt.Printf("Stream %d achieved rate: %.2f Mbps\n", k, achievedRate)

		// // time.Sleep(10 * time.Second)
		// nextwaittime := time.Now().Add(10 * time.Second)
		// // for time.Now().Before(nextwaittime) {
		// // 	// Actively wait without sleeping
		// // }
		// // for time.Until(nextwaittime) > 0 {
		// // 	runtime.Gosched() // Yield the processor to improve timing accuracy
		// // }
		time.Sleep(1 * time.Second)
	}

	writer.Write(achievedRates)
	fmt.Println("Achieved rates written to CSV!")

	writer.Flush()

	// Wait to ensure all streams finish before exiting
	time.Sleep(3 * time.Second)
}
