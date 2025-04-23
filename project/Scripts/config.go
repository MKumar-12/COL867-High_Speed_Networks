package main

import (
	"fmt"
	"time"
	"os"
)

// Configuration constants for the UDP receiver
// and TCP sender
const (
	ReceiverIP = "10.0.0.2"
	ListenIP   = "10.0.0.2"
	UDPPort    = 5005
	TCPPort    = 6000
	PacketSize = 1500
	NumPackets = 100
	DeltaRK    = 20
	BufferSize = 2048
	Timeout    = 2 * time.Second
	DATA_DIR = "../Data_D2"
)


func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: go run config.go [sender|receiver]")
        return
    }

    role := os.Args[1]
    switch role {
    case "sender":
        runSender()
    case "receiver":
        runReceiver()
    default:
        fmt.Println("Invalid role. Use 'sender' or 'receiver'")
    }
}