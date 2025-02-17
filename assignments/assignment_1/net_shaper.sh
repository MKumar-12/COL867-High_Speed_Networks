#!/bin/bash


# Network Download Shaper using tc and IFB
# ----------------------------------------
# This script limits the download bandwidth on a given network interface.
# It uses a virtual interface (ifb0) to redirect and shape incoming traffic.



# Usage:
#   ./net_shaper.sh start <interface> <bandwidth>  # Start shaping
#   ./net_shaper.sh update <interface> <bandwidth> # Update bandwidth
#   ./net_shaper.sh stop <interface>               # Stop shaping
#   ./net_shaper.sh list <interface>               # Show current shaping status
#

# Examples:
#   ./net_shaper.sh start eth0 5mbit
#   ./net_shaper.sh update eth0 2mbit
#   ./net_shaper.sh stop eth0
#   ./net_shaper.sh list eth0

# Function to check if IFB module is loaded
load_ifb_module() {
    if ! lsmod | grep -q ifb; then
        sudo modprobe ifb
    fi
}

# Function to start traffic shaping for download
start_shaping() {
    local interface=$1
    local bandwidth=$2
    local vinterface="ifb0"  # Virtual interface

    # Ensure the interface exists
    if ! ip link show "$interface" &>/dev/null; then
        echo "Error: Interface $interface not found!"
        exit 1
    fi

    # Load IFB module if not loaded
    load_ifb_module

    # Set up IFB interface
    sudo ip link set dev "$vinterface" up

    # Redirect ingress (download) traffic to IFB
    sudo tc qdisc add dev "$interface" handle ffff: ingress
    sudo tc filter add dev "$interface" parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev "$vinterface"

    # Apply shaping on IFB
    sudo tc qdisc add dev "$vinterface" root handle 1: htb default 10
    sudo tc class add dev "$vinterface" parent 1: classid 1:10 htb rate "$bandwidth"

    echo "Download shaping started: $interface limited to $bandwidth"
}

# Function to update download bandwidth limit
update_shaping() {
    local interface=$1
    local new_bandwidth=$2
    local vinterface="ifb0"

    # Ensure the virtual interface is shaped
    if ! tc qdisc show dev "$vinterface" | grep -q "htb"; then
        echo "Error: No shaping found on $interface. Use 'start' first."
        exit 1
    fi

    # Update the bandwidth
    sudo tc class change dev "$vinterface" parent 1: classid 1:10 htb rate "$new_bandwidth"
    echo "Updated download bandwidth on $interface to $new_bandwidth"
}

# Function to stop download shaping
stop_shaping() {
    local interface=$1
    local vinterface="ifb0"

    # Remove traffic control rules
    sudo tc qdisc del dev "$interface" ingress
    sudo tc qdisc del dev "$vinterface" root

    echo "Download shaping removed from $interface"
}

# Function to list shaping status
list_shaping() {
    local interface=$1
    local vinterface="ifb0"
    echo "Current shaping status for $interface:"
    sudo tc qdisc show dev "$interface"
    sudo tc qdisc show dev "$vinterface"
    sudo tc class show dev "$vinterface"
}

# Main script logic
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {start|update|stop|list} <interface> [bandwidth]"
    exit 1
fi

command=$1
interface=$2
bandwidth=$3

case "$command" in
    start)
        if [[ -z "$bandwidth" ]]; then
            echo "Error: Missing bandwidth for 'start'"
            exit 1
        fi
        start_shaping "$interface" "$bandwidth"
        ;;
    update)
        if [[ -z "$bandwidth" ]]; then
            echo "Error: Missing new bandwidth for 'update'"
            exit 1
        fi
        update_shaping "$interface" "$bandwidth"
        ;;
    stop)
        stop_shaping "$interface"
        ;;
    list)
        list_shaping "$interface"
        ;;
    *)
        echo "Invalid command. Use start, update, stop, or list."
        exit 1
        ;;
esac

