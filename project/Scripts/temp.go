package main

import (
	"encoding/csv"
	"os"
	"fmt"
	"net"
	"time"
	"syscall"
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
	Timeout    = 10 * time.Second
)

// func setRealtimePriority() {
// 	if runtime.GOOS == "linux" {
// 		param := &unix.SchedParam{Priority: 99}
// 		err := unix.SchedSetscheduler(0, unix.SCHED_FIFO, param)
// 		if err != nil {
// 			fmt.Println("Failed to set real-time priority:", err)
// 		}
// 	} else {
// 		fmt.Println("Real-time scheduling not supported on this OS.")
// 	}
// }

func main() {
	// setRealtimePriority()

	// Create UDP socket for receiving
	err := syscall.Setpriority(syscall.PRIO_PROCESS, 0, -5)
	if err != nil {
		fmt.Println("Failed to set priority:", err)
	} else {
		fmt.Println("Receiver priority set to -5")
	}

	udpAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", ListenIP, UDPPort))
	if err != nil {
		fmt.Println("Error resolving UDP address:", err)
		return
	}

	udpConn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		fmt.Println("Error creating UDP socket:", err)
		return
	}
	defer udpConn.Close()

	// udpConn.SetReadBuffer(1024 * 1024)

	// Get the file descriptor for the socket
	fd, err := udpConn.File()
	if err != nil {
		fmt.Println("Error getting file descriptor:", err)
		return
	}
	defer fd.Close()
	// Reduce OS receive buffer to 8 KB (default is often 208 KB or more)
	err = syscall.SetsockoptInt(int(fd.Fd()), syscall.SOL_SOCKET, syscall.SO_RCVBUF, 512*1024)
	if err != nil {
		fmt.Println("Error setting SO_RCVBUF:", err)
	}
	// Enable SO_REUSEPORT to allow multiple sockets on the same port
	const SO_REUSEPORT = 15
	err = syscall.SetsockoptInt(int(fd.Fd()), syscall.SOL_SOCKET, SO_REUSEPORT, 1)
	if err != nil {
		fmt.Println("Error setting SO_REUSEPORT:", err)
	}

//	udpConn.SetReadDeadline(time.Now().Add(100 * time.Millisecond))

	fmt.Println("UDP receiver is running on port 5000...")

	// Create TCP listener
	tcpAddr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("%s:%d", ListenIP, TCPPort))
	if err != nil {
		fmt.Println("Error resolving TCP address:", err)
		return
	}

	tcpListener, err := net.ListenTCP("tcp", tcpAddr)
	if err != nil {
		fmt.Println("Error starting TCP server:", err)
		return
	}
	defer tcpListener.Close()

	fmt.Println("TCP server is running. Waiting for sender to connect...")

	// Wait for a TCP connection before proceeding
	tcpConn, err := tcpListener.Accept()
	if err != nil {
		fmt.Println("Error accepting TCP connection:", err)
		return
	}
	defer tcpConn.Close()

	fmt.Println("TCP connection established. Now waiting for UDP probing packets...")

	file, err := os.OpenFile("stream_rates_rout.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		fmt.Println("Error opening CSV file:", err)
		return
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()


	var receivedRates []string

	// Receive UDP packets
	buffer := make([]byte, BufferSize)
	receivedBytes := 0
	startTime := time.Now()
	lastPacketTime := time.Now()
	packetCount := 0
	// arrivalTimes := []time.Time{} // UPDATED: Store arrival times
	var totalInterArrival float64

	for {
		// udpConn.SetReadDeadline(time.Now().Add(Timeout))
		n, _, err := udpConn.ReadFromUDP(buffer)
		if err != nil {
			fmt.Println("Timeout reached! Stopping UDP reception.")
			break
		}

		receivedBytes += n
		packetCount++
		// fmt.Printf("Received packet %d, Total bytes received: %d\n", packetCount, n)
		lastPacketTime = time.Now()
		// arrivalTimes = append(arrivalTimes, time.Now())

		if packetCount >= NumPackets {
			fmt.Println("All burst packets received.")
			break
		}
	}

	// Compute r_out
	fmt.Println("start time= ", startTime)
	endTime := lastPacketTime
	fmt.Println("end time= ", endTime)

	duration := endTime.Sub(startTime).Seconds()
	fmt.Println("duration = ", duration)
	fmt.Println("load received in bits= ", receivedBytes*8)

	// for i := 1; i < len(arrivalTimes); i++ {
	// 	totalInterArrival += arrivalTimes[i].Sub(arrivalTimes[i-1]).Seconds()
	// }

	// avgInterArrival := totalInterArrival / float64(len(arrivalTimes)-1)
	// rOut := (float64(PacketSize*8) / (avgInterArrival * 1e6))
/*	rOut := 0.0
	if duration > 0 {
		rOut = float64(receivedBytes*8) / (duration * 1e6)
	}
	fmt.Printf("Estimated r_out: %.2f mbps\n", rOut)

	// Compute delta_r
	deltaR := rOut / DeltaRK
	if deltaR == 0 {
	duration := endTime.Sub(startTime).Seconds()
	fmt.Println("duration = ", duration)
	fmt.Println("load received in bits= ", receivedBytes*8)

	// for i := 1; i < len(arrivalTimes); i++ {
	// 	totalInterArrival += arrivalTimes[i].Sub(arrivalTimes[i-1]).Seconds()
	// }

	// avgInterArrival := totalInterArrival / float64(len(arrivalTimes)-1)
	// rOut := (float64(PacketSize*8) / (avgInterArrival * 1e6))*/

	rOut := 0.0
	if duration > 0 {
		rOut = float64(receivedBytes*8) / (duration * 1e6)
	}
	fmt.Printf("Estimated r_out: %.2f mbps\n", rOut)

	// Compute delta_r
	deltaR := rOut / DeltaRK
	if deltaR == 0 {
		deltaR = 1e6 // Default to 1 Mbps if no packets received
	}
	fmt.Printf("Computed delta_r: %.2f mbps\n", deltaR)

	receivedRates = append(receivedRates, fmt.Sprintf("%.2f", rOut))

	_, err = fmt.Fprintf(tcpConn, "%.2f", deltaR)
	if err != nil {
		fmt.Println("Error sending delta_r:", err)
	}

	// Now receive 20 streams
	fmt.Println("Waiting for 20 streams...")
	for k := 1; k <= DeltaRK; k++ {
		//streamRate := float64(k) * deltaR                   // Rate in Mb
		//inter := float64(PacketSize*8) / (streamRate * 1e6) // Interval in seconds
		//intervalNs := time.Duration((inter) * 1e9)            // Convert to nanoseconds

		// Wait for TCP message signaling stream start
		buffer := make([]byte, BufferSize)
		_, err := tcpConn.Read(buffer)
		if err != nil {
			fmt.Printf("Error reading stream %d start signal: %v\n", k, err)
			return
		}

		// buffer := make([]byte, 1024)
		// n, err := conn.Read(buffer)
		// if err == nil {
		// 	fmt.Printf("Received message: %s\n", string(buffer[:n]))
		// 	tcpConn.Write([]byte("ACK"))
		// }
		// fmt.Printf("Stream %d start signal received. Now receiving UDP packets...\n", k)

		streamBytes := 0
		// streamStart := time.Now()
		streamPackets := 0
		// udpConn.SetReadDeadline(time.Now().Add(Timeout))

		streamArrivalTimes := []time.Time{}
		// streamStart := time.Now()
		for {
			timestart := time.Now()

			n, _, err := udpConn.ReadFromUDP(buffer)
			if err != nil {
				fmt.Println("time taken for timeout = ", time.Now().Sub(timestart).Seconds())
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

			if streamPackets >= NumPackets {
				fmt.Println("All stream packets received.")
				break
			}
			/*nextSendTime := streamStart.Add(time.Duration(streamPackets+1) * intervalNs)
			for time.Since(nextSendTime) < 0 {
				// Empty loop for precise wait
			}*/
		}

		// Compute r_out for this stream
		// streamEnd := time.Now()
		// streamDuration := streamEnd.Sub(streamStart).Seconds()
		// streamRout := 0.0
		// if streamDuration > 0 {
		// 	streamRout = float64(streamBytes*8) / (streamDuration * 1e6)
		// }
		// fmt.Printf("Stream %d: Estimated r_out: %.2f mbps\n", k, streamRout)

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

		// time.Sleep(5 * time.Second)
		// nextwaittime := time.Now().Add(5 * time.Second)

		// for time.Since(nextwaittime) < 0 {
		// 	// Empty loop for precise wait
		// }

	}
	writer.Write(receivedRates)
	fmt.Println("Achieved rates written to CSV!")

}

