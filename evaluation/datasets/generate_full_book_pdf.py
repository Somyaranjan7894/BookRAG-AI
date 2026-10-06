"""Generator for realistic 100+ page technical book PDF for BookRAG Phase 23 Full-Book Benchmarking.

Creates:
'Foundations of Distributed Systems, Cloud Architecture, and Machine Learning Infrastructure'
(105 pages across 10 technical chapters).
"""

from pathlib import Path
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def create_full_book_pdf(output_path: Path) -> Path:
    """Generate a realistic 105-page technical book PDF using PyMuPDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # Chapter outline across 105 pages:
    # Page 1: Title page
    # Page 2: Table of Contents & Preface
    # Pages 3-12: Chapter 1 - Distributed Systems Foundations & Time
    # Pages 13-22: Chapter 2 - Consensus Protocols & Fault Tolerance (Raft, Paxos, Quorums)
    # Pages 23-32: Chapter 3 - Storage Architectures, LSM-Trees & Partitioning
    # Pages 33-42: Chapter 4 - Distributed Transactions & Concurrency Control (2PC, 3PC, MVCC, Spanner)
    # Pages 43-52: Chapter 5 - Stream Processing & Asynchronous Event Systems (Kafka, Flink, Watermarks)
    # Pages 53-62: Chapter 6 - Vector Databases & High-Dimensional Similarity Search (HNSW, IVF, PQ)
    # Pages 63-72: Chapter 7 - Neural Information Retrieval & Cross-Encoder Reranking
    # Pages 73-82: Chapter 8 - Large Language Model Serving & KV Cache Optimization (PagedAttention, vLLM)
    # Pages 83-92: Chapter 9 - Distributed Caching & Memory Consistency (Eviction, Cache Stampede, Redlock)
    # Pages 93-105: Chapter 10 - Site Reliability Engineering, Observability & Security (Four Golden Signals, Zero Trust)

    chapter_data = {
        1: {
            "title": "Distributed Systems Foundations & Logical Clocks",
            "pages": {
                3: (
                    "1.1 Models of Distributed Computation and Network Synchrony",
                    "A distributed system consists of autonomous computing nodes that communicate across an imperfect network "
                    "lacking a shared global physical clock or shared physical memory. Distributed execution models are classified into "
                    "three distinct synchrony categories: synchronous networks, asynchronous networks, and partially synchronous networks. "
                    "In a synchronous network model, message transmission delay has an absolute deterministic upper bound delta, and processor "
                    "step execution times are bounded by constant epsilon. In an asynchronous network model, message transmission delays "
                    "and processor step speeds are completely unbounded, meaning no assumptions can be made regarding timeout thresholds. "
                    "The partially synchronous network model establishes that communication behaves asynchronously up to an unknown Global "
                    "Stabilization Time (GST), after which deterministic delay bounds hold. Real-world datacenter and WAN networks are modeled "
                    "as partially synchronous environments, as temporary congestion, fiber cuts, and routing flaps violate strict synchrony.",
                ),
                4: (
                    "1.2 The FLP Impossibility Result and System Implications",
                    "The Fischer-Lynch-Paterson (FLP) theorem demonstrates that in an asynchronous network model, no deterministic consensus protocol "
                    "can guarantee both safety and liveness in the presence of even a single unannounced crash failure. The FLP theorem establishes "
                    "that every consensus protocol must possess an initial bivalent execution state from which both 0 and 1 decision values remain "
                    "reachable. Because message delays are unbounded, a node cannot distinguish between a crashed peer and a peer operating over a "
                    "severely delayed network link. Modern distributed systems circumvent FLP impossibility by employing randomized consensus algorithms, "
                    "relying on partial synchrony assumptions with heartbeat timeouts, or introducing failure detector abstractions with completeness "
                    "and accuracy properties.",
                ),
                5: (
                    "1.3 Lamport Timestamps and Causal Ordering",
                    "Leslie Lamport introduced logical scalar clocks to establish a partial causal order of events in distributed systems without "
                    "relying on synchronized physical hardware clocks. The happens-before relation, denoted as a -> b, satisfies three conditions: "
                    "first, if events a and b occur on the same process and a precedes b, then a -> b; second, if a is the sending of a message "
                    "and b is the receipt of that same message, then a -> b; third, if a -> b and b -> c, then a -> c by transitivity. "
                    "Each node maintains a monotonically increasing integer counter C. Upon executing an internal event, the node increments C by 1. "
                    "When sending a message, the current counter C is attached as metadata. Upon receiving a message with timestamp T, the recipient "
                    "updates its local counter according to the deterministic rule: C = max(C, T) + 1. While Lamport timestamps guarantee that if a -> b "
                    "then C(a) < C(b), the converse does not hold, as two independent concurrent events may have C(a) < C(b) without causal dependency.",
                ),
                6: (
                    "1.4 Vector Clocks and Causality Concurrency Detection",
                    "Vector clocks extend scalar logical clocks to provide isomorphic characterization of causal relationships between distributed events. "
                    "In a cluster of N participating nodes, each node maintains a vector V of size N, where component V[i] represents node i's "
                    "knowledge of events initiated by node i. Upon local event occurrence at node i, V[i] is incremented by 1. When transmitting a message, "
                    "node i attaches its entire vector clock. Upon receiving vector V_msg, node i updates every component using the element-wise maximum: "
                    "V[j] = max(V[j], V_msg[j]) for all j, followed by incrementing V[i] by 1. Two events a and b are defined as causally dependent if one "
                    "strictly dominates the other; otherwise, if neither dominates, the events are mathematically concurrent, denoted as a || b. "
                    "Vector clocks are used in distributed storage systems such as Amazon Dynamo and Riak to track conflicting object updates.",
                ),
                7: (
                    "1.5 State Machine Replication Architecture",
                    "State Machine Replication (SMR) is the foundational architecture for building fault-tolerant distributed services. "
                    "A deterministic state machine begins in an initial state S_0 and transitions to state S_{t+1} = f(S_t, m) upon processing input command m. "
                    "If multiple independent replicas initialize in the identical state S_0 and process an identical sequence of input commands "
                    "[m_1, m_2, ..., m_k] in strictly the same total order, all non-faulty replicas will arrive at identical final states and produce "
                    "identical outputs. Total order broadcast is mathematically equivalent to distributed consensus, requiring replicas to agree on the "
                    "precise command slot assignments in an immutable distributed commit log.",
                ),
                8: (
                    "1.6 Network Partitions and Split-Brain Scenarios",
                    "A network partition occurs when communication links fail such that nodes are separated into two or more disjoint connected components. "
                    "If a distributed system continues accepting write operations in multiple disconnected partitions without consensus verification, "
                    "the system experiences a split-brain condition. Split-brain execution results in divergent, irreconcilable database state, silent "
                    "data corruption, and inconsistent responses to clients. Quorum-based consensus protocols prevent split-brain by requiring that any valid "
                    "write or state transition must obtain explicit authorization from a strict majority quorum of nodes, ensuring that at most one partition "
                    "can ever satisfy the quorum requirement.",
                ),
                9: (
                    "1.7 Physical Clock Drift and Precision Time Protocol",
                    "Physical quartz crystal oscillators in server hardware experience thermal frequency drift, typically drifting by 10 to 100 parts "
                    "per million (ppm), resulting in clock divergence of up to 8.6 seconds per day without synchronization. Network Time Protocol (NTP) "
                    "synchronizes physical clocks across wide-area networks over UDP port 123, typically achieving accuracy within 10 to 50 milliseconds. "
                    "In local datacenters, the Precision Time Protocol (PTP, IEEE 1588) utilizes hardware packet timestamping at the Network Interface Card (NIC) "
                    "PHY layer to achieve sub-microsecond clock synchronization accuracy, enabling precise distributed event sequencing.",
                ),
                10: (
                    "1.8 Failure Detectors and Heartbeat Intervals",
                    "Unreliable failure detectors monitor node health using periodic heartbeat exchanges. A failure detector is characterized by two "
                    "orthogonal completeness and accuracy properties: strong completeness requires that every crashed process is eventually suspected by "
                    "every correct process; strong accuracy requires that no correct process is ever suspected before crashing. The phi-accrual failure detector "
                    "replaces binary healthy/crashed states with a continuous suspicion scale phi = -log10(P_later(t - t_last)), where P_later is the probability "
                    "that a heartbeat arrives later than elapsed time t. This adaptive probabilistic model accommodates unpredictable WAN network jitter.",
                ),
                11: (
                    "1.9 Gossip Protocols and Epidemic Dissemination",
                    "Gossip protocols provide decentralized, fault-tolerant state dissemination inspired by epidemic spread models. In each round of length T, "
                    "every node selects k peer nodes uniformly at random from its membership table and transmits its current state or anti-entropy digest. "
                    "The number of informed nodes grows exponentially during the initial infection phase and logarithmically approaches total network coverage. "
                    "For a cluster of N nodes, gossip protocols achieve complete cluster convergence in O(log N) time rounds with high probability, while "
                    "generating message traffic of O(N log N). Gossip is widely utilized in Apache Cassandra and Consul for node discovery and failure detection.",
                ),
                12: (
                    "1.10 Chapter Summary and Review Questions",
                    "Chapter 1 established foundational distributed computing principles: synchrony models (synchronous, asynchronous, partially synchronous), "
                    "the FLP impossibility result, logical ordering via Lamport timestamps and vector clocks, state machine replication, split-brain hazards, "
                    "and failure detectors. System architects must recognize that unbounded asynchronous networks preclude perfect failure detection and "
                    "that consensus requires trade-offs between availability and consistency under network partitions.",
                ),
            },
        },
        2: {
            "title": "Consensus Protocols & Fault Tolerance",
            "pages": {
                13: (
                    "2.1 Principles of Quorum Consensus Systems",
                    "A quorum consensus system requires that operations on distributed shared state intersect across intersecting node subsets. "
                    "In a cluster of N independent replica nodes, a read quorum of size R and a write quorum of size W must satisfy the fundamental "
                    "pigeonhole condition: R + W > N. When R + W > N, any read quorum is guaranteed to contain at least one node that participated in the "
                    "most recent write quorum, ensuring strong read-your-writes consistency when combined with monotonic version timestamps. "
                    "For symmetric majority quorum systems, both read and write quorums are chosen such that Q = floor(N / 2) + 1. In a 5-node cluster, "
                    "the strict majority quorum size is exactly 3 nodes, allowing the cluster to tolerate up to 2 simultaneous node crashes without loss of availability.",
                ),
                14: (
                    "2.2 Paxos Protocol Foundations: The Single-Decree Synod",
                    "Leslie Lamport's Paxos consensus algorithm guarantees safety under arbitrary message loss, reordering, and duplication in asynchronous networks. "
                    "The Single-Decree Synod protocol reaches agreement on a single proposed value across two phases: Phase 1 (Prepare/Promise) and "
                    "Phase 2 (Accept/Accepted). In Phase 1a, a proposer chooses a globally unique, monotonically increasing proposal number n and broadcasts "
                    "Prepare(n) to acceptors. In Phase 1b, each acceptor promises not to accept proposals numbered less than n; if the acceptor previously "
                    "accepted a proposal, it returns the value v and proposal number n_v of its highest accepted proposal. In Phase 2a, once the proposer receives "
                    "promises from a majority of acceptors, it selects value v (the value with the highest n_v from Phase 1b, or its own value if none), and broadcasts "
                    "Accept(n, v). In Phase 2b, an acceptor accepts proposal (n, v) if it has not promised to a proposal higher than n.",
                ),
                15: (
                    "2.3 Multi-Paxos and Stable Leader Optimization",
                    "Executing full two-phase Paxos for every individual log entry incurs two round-trip network latencies (4 communication hops). Multi-Paxos "
                    "optimizes common-case execution by amortizing Phase 1 across a stream of log entries. A stable leader is elected by executing Phase 1 once "
                    "for all future log instances up to infinity. Once a leader secures promises for all unbounded future slots, subsequent client commands "
                    "only require Phase 2 (Accept and Accepted), reducing steady-state commit latency to exactly one network round-trip. If leader failure occurs, "
                    "a successor node must re-execute Phase 1 to discover previously proposed values and establish a new stable lease.",
                ),
                16: (
                    "2.4 The Raft Consensus Algorithm: Architecture and Roles",
                    "The Raft consensus algorithm was formulated by Ongaro and Ousterhout to provide equivalent fault-tolerant state machine replication "
                    "to Multi-Paxos with dramatically superior understandability and formal decomposability. A Raft cluster decomposes consensus into "
                    "three distinct subproblems: leader election, log replication, and safety enforcement. At any moment, each node resides in one of three "
                    "mutually exclusive states: Leader, Follower, or Candidate. Time in Raft is divided into arbitrary terms identified by contiguous integers. "
                    "Each term begins with a leader election; if the election succeeds, that leader directs log replication for the remainder of the term.",
                ),
                17: (
                    "2.5 Raft Leader Election and Randomized Election Timeouts",
                    "Nodes initialize in the Follower state. If a follower receives no heartbeat AppendEntries RPC within a randomized election timeout "
                    "between 150 milliseconds and 300 milliseconds, it assumes the leader has crashed, increments its currentTerm counter, transitions to "
                    "Candidate state, votes for itself, and broadcasts RequestVote RPCs to all peers. Randomized timeouts prevent split votes by ensuring "
                    "that one candidate's timer expires significantly earlier than its peers, enabling it to collect majority votes before competing elections "
                    "start. A candidate wins the election if it receives affirmative votes from a strict majority (floor(N / 2) + 1) of cluster nodes.",
                ),
                18: (
                    "2.6 Raft Log Invariant and Commit Rules",
                    "Raft enforces two fundamental log invariants: first, if two entries in different logs have the same index and term, they store the "
                    "identical command; second, if two entries in different logs have the same index and term, then their logs are identical in all preceding "
                    "entries. A log entry is committed once the leader that created the entry replicates it on a majority of nodes. To prevent subtle "
                    "re-election overwrites, Raft leaders never commit an entry from a previous term by counting replicas; leaders only commit entries from "
                    "their current term by counting replicas, which indirectly commits earlier entries by virtue of the Log Matching Property.",
                ),
                19: (
                    "2.7 Raft Cluster Membership Changes: Joint Consensus",
                    "Modifying cluster membership dynamically (adding or removing nodes) introduces the danger that two independent majorities could form "
                    "simultaneously under the old and new configurations. Raft resolves this using a two-phase Joint Consensus approach. The leader proposes "
                    "a transitional configuration entry C_old,new. In joint consensus, log entries must be committed by majorities of both C_old and C_new "
                    "independently. Once C_old,new is committed to both configurations, the leader proposes the final configuration C_new, which only requires "
                    "a majority of C_new. Single-server membership changes allow atomic transitions without full joint consensus.",
                ),
                20: (
                    "2.8 Byzantine Fault Tolerance and PBFT Protocol",
                    "Crash fault tolerant (CFT) protocols like Paxos and Raft assume nodes fail by halting or network silence. In adversarial or untrusted "
                    "environments, Byzantine faults allow nodes to behave arbitrarily, including transmitting conflicting messages to different peers. "
                    "Castro and Liskov's Practical Byzantine Fault Tolerance (PBFT) protocol guarantees safety in asynchronous networks containing up to f "
                    "Byzantine nodes, requiring a minimum cluster size of N >= 3f + 1 nodes. PBFT operates across three phases: Pre-Prepare, Prepare, and "
                    "Commit, utilizing cryptographic digital signatures and two-thirds supermajority quorums (2f + 1 nodes) to achieve deterministic consensus.",
                ),
                21: (
                    "2.9 Leader Leases and Linearizable Read Optimization",
                    "In basic consensus protocols, processing a read request requires the leader to replicate a dummy log entry through quorum consensus "
                    "to confirm that it has not been deposed by a network partition. To achieve sub-millisecond read latency, leaders utilize bounded "
                    "leader leases. Followers promise not to elect a new leader for a duration of T_lease following their last vote or heartbeat response. "
                    "So long as the leader's local clock has advanced by less than T_lease / (1 + clock_drift), the leader can serve linearizable reads directly "
                    "from its local state machine memory without executing network round-trips. Clock drift must be strictly bounded to prevent stale reads.",
                ),
                22: (
                    "2.10 Chapter Summary: Paxos vs Raft Trade-Offs",
                    "Both Paxos and Raft provide provable safety under partial synchrony, tolerating floor((N - 1) / 2) crash failures. While Multi-Paxos "
                    "permits out-of-order log proposal commits, Raft requires strict sequential log appending, which simplifies state machine recovery "
                    "and log compaction at the cost of minor throughput pipelining flexibility. In production infrastructure, Raft is the engine behind "
                    "etcd and HashiCorp Consul, while Multi-Paxos variants power Google Chubby and Apache ZooKeeper (Zab protocol).",
                ),
            },
        },
        3: {
            "title": "Storage Architectures, LSM-Trees & Partitioning",
            "pages": {
                23: (
                    "3.1 Storage Engine Foundations: In-Place Updates vs Append-Only",
                    "Database storage engines balance write amplification, read amplification, and space amplification. Traditional relational databases "
                    "employ page-based B+ trees that perform in-place updates on fixed-size disk blocks (typically 4 KB, 8 KB, or 16 KB). While in-place "
                    "updates provide fast O(log N) point reads, random write workloads suffer severe I/O bottlenecks because modifying a single record "
                    "requires reading, updating, and writing back full 8 KB disk pages. In contrast, append-only log-structured storage engines convert "
                    "all writes, updates, and deletes into sequential append operations, maximizing write throughput on solid-state drives and hard disks.",
                ),
                24: (
                    "3.2 Log-Structured Merge-Tree (LSM-Tree) Architecture",
                    "The Log-Structured Merge-Tree (LSM-Tree) architecture decouples incoming write operations from persistent disk organization. An LSM-tree "
                    "consists of three core components: an in-memory sorted write buffer termed the MemTable (frequently implemented as a SkipList or Red-Black Tree), "
                    "a disk-based sequential Write-Ahead Log (WAL) ensuring durability across power outages, and immutable on-disk Sorted String Table (SSTable) files "
                    "structured into hierarchical levels (Level 0, Level 1, ..., Level k). Incoming writes append immediately to the WAL and insert into the MemTable. "
                    "Because MemTable insertions occur in RAM, write latency is sub-microsecond.",
                ),
                25: (
                    "3.3 Sorted String Table (SSTable) Anatomy and Bloom Filters",
                    "An SSTable is an immutable disk file storing key-value pairs sorted lexicographically by key. An SSTable file comprises sequential data blocks, "
                    "a sparse two-level block index mapping keys to byte offsets, and a probabilistic Bloom filter. When executing a point lookup for key K, the storage "
                    "engine first queries the Bloom filter. If the Bloom filter returns false, the key is guaranteed absent from the SSTable, eliminating an expensive "
                    "disk seek with zero false negatives. If the Bloom filter returns true, binary search over the sparse block index identifies the exact 4 KB data block "
                    "containing K, performing at most one disk read.",
                ),
                26: (
                    "3.4 LSM Compaction Strategies: Size-Tiered vs Leveled Compaction",
                    "Because updates and deletions create obsolete versions in newer SSTables, background compaction merges older SSTables to reclaim disk space. "
                    "In Size-Tiered Compaction, multiple SSTables of approximately equal size are merged into a single larger SSTable once a threshold count is reached; "
                    "this yields low write amplification but high space amplification (up to 100% temporary overhead). In Leveled Compaction (employed by RocksDB), "
                    "each level has a strict capacity limit (e.g., Level 1 = 10 MB, Level 2 = 100 MB, Level 3 = 1 GB), and keys within any level above Level 0 are "
                    "strictly non-overlapping. Leveled compaction provides predictable read performance and minimal space amplification at the expense of higher write amplification.",
                ),
                27: (
                    "3.5 B+ Tree Indexing and Write-Ahead Logging",
                    "A B+ tree is a self-balancing N-ary search tree optimized for block-oriented storage systems. All user data records are stored exclusively in "
                    "leaf pages linked sequentially by doubly-linked pointers for efficient range scanning, while interior pages store routing keys and child page pointers. "
                    "A B+ tree with fan-out factor B = 100 and depth D = 4 can index 100 million records with at most 4 page traversals. To guarantee durability "
                    "and atomicity against mid-write crashes, B+ tree engines write physiological redo/undo log records to a WAL before modifying dirty buffer pool pages, "
                    "following the Write-Ahead Logging protocol.",
                ),
                28: (
                    "3.6 Consistent Hashing and Hash Ring Topologies",
                    "Horizontal data partitioning (sharding) distributes keys across a cluster of storage nodes. Simple hash partitioning using key mod N fails "
                    "under dynamic scaling because adding or removing a single node invalidates nearly all key placements (N / (N + 1) redistribution). Karger et al. "
                    "introduced Consistent Hashing, which maps both node identifiers and data keys onto a continuous circular hash ring of size 2^128 - 1 using cryptographic "
                    "hashes (such as MD5 or MurmurHash3). A key is assigned to the first node encountered by traversing clockwise along the perimeter of the hash ring.",
                ),
                29: (
                    "3.7 Virtual Nodes and Non-Uniform Load Balancing",
                    "Basic consistent hashing with physical nodes produces non-uniform key distributions and hot-spots due to stochastic clustering of hash points. "
                    "Distributed storage systems introduce Virtual Nodes (Vnodes), where each physical server is mapped to V distinct virtual positions on the hash ring "
                    "(typically V = 128 to 256). Distributing multiple virtual points per physical machine smooths statistical variance across the ring perimeter, "
                    "achieving near-perfect uniform data distribution within 1% to 3% deviation. When a physical server is added or decommissioned, load is transferred "
                    "fractionally and concurrently across all surviving peers rather than overwhelming an adjacent neighbor.",
                ),
                30: (
                    "3.8 Range Partitioning vs Hash Partitioning",
                    "Range partitioning divides data by contiguous key intervals (e.g., keys A through F on node 1, G through M on node 2). Range partitioning "
                    "enables efficient sequential range scans because related records reside on contiguous nodes; however, monotonically increasing keys (such as timestamps "
                    "or auto-incrementing IDs) create severe write hot-spots where all traffic routes exclusively to the tail partition. Hash partitioning distributes keys "
                    "uniformly across nodes by applying cryptographic hash functions, eliminating write hot-spots at the cost of requiring scatter-gather distributed queries "
                    "for range scans.",
                ),
                31: (
                    "3.9 Storage Replication: Synchronous vs Asynchronous",
                    "Distributed storage achieves fault tolerance through data replication. In synchronous replication, the primary node waits for confirmation from all "
                    "replica nodes before returning a successful write acknowledgment to the client; this guarantees zero data loss (Recovery Point Objective RPO = 0) "
                    "at the cost of write latency bounded by the slowest replica node. In asynchronous replication, the primary acknowledges writes immediately after local "
                    "commit and propagates updates in background streams; this maximizes write throughput and minimizes client latency, but network or hardware failure "
                    "prior to replication creates data loss windows.",
                ),
                32: (
                    "3.10 Chapter Summary and Architecture Trade-Offs",
                    "Chapter 3 contrasted storage engine internals and horizontal partitioning mechanisms. LSM-trees optimize write-heavy ingestion workloads with sequential "
                    "WAL appends and leveled SSTable compaction, whereas B+ trees optimize point reads and range scans with in-place page updates. Horizontal scaling relies "
                    "on consistent hashing with virtual nodes to achieve uniform partition balancing without catastrophic data shuffling during cluster membership changes.",
                ),
            },
        },
        4: {
            "title": "Distributed Transactions & Concurrency Control",
            "pages": {
                33: (
                    "4.1 ACID Properties in Distributed Relational Systems",
                    "A transaction represents an atomic sequence of read and write operations executing on database state. In distributed environments, "
                    "ACID properties must hold across multiple independent network nodes: Atomicity guarantees that all sub-operations commit or all abort; "
                    "Consistency enforces schema invariants and integrity constraints; Isolation ensures concurrent transactions execute without interference "
                    "as if executing sequentially; Durability guarantees committed modifications persist permanently despite server crashes. "
                    "Providing ACID guarantees across network partitions introduces substantial communication overhead and synchronization delays.",
                ),
                34: (
                    "4.2 The Two-Phase Commit (2PC) Protocol",
                    "The Two-Phase Commit (2PC) protocol coordinates atomic transaction commits across distributed resource managers. 2PC involves a designated "
                    "Transaction Coordinator (TC) and multiple Participant nodes (cohorts). In Phase 1 (Prepare Phase), the coordinator broadcasts a Prepare message "
                    "to all participants. Each participant executes transaction operations locally up to commit, writes undo/redo logs, and returns a vote of either VOTE_COMMIT "
                    "or VOTE_ABORT. In Phase 2 (Commit Phase), if all participants voted VOTE_COMMIT, the coordinator writes a global commit record to its WAL and broadcasts "
                    "Global_Commit; if any participant voted VOTE_ABORT or timed out, the coordinator broadcasts Global_Abort. All participants execute the decision and acknowledge.",
                ),
                35: (
                    "4.3 Coordinator Failure Modes and 2PC Blocking Hazards",
                    "The critical architectural vulnerability of Two-Phase Commit is that it is an inherently blocking protocol. If the Transaction Coordinator "
                    "crashes after participants have responded with VOTE_COMMIT in Phase 1 but before broadcasting Global_Commit or Global_Abort in Phase 2, "
                    "participants enter an uncertain (in-doubt) state. In this uncertain state, participants hold active row and table locks on local resources "
                    "and cannot independently abort (because the coordinator may have decided to commit) nor commit (because another participant may have voted abort). "
                    "These locks remain indefinitely blocked until the coordinator recovers, causing cascading throughput collapse in high-concurrency systems.",
                ),
                36: (
                    "4.4 The Three-Phase Commit (3PC) Protocol",
                    "The Three-Phase Commit (3PC) protocol was designed by Skeen to eliminate the blocking vulnerability of 2PC under fail-stop coordinator crashes. "
                    "3PC divides the commit sequence into three non-blocking phases: Can-Commit?, Pre-Commit, and Do-Commit. By inserting an intermediate Pre-Commit "
                    "state, 3PC guarantees that if any participant has entered the Pre-Commit state, all participants have agreed to commit. If coordinator failure occurs, "
                    "an election protocol can query surviving participants and safely progress the transaction without blocking. However, 3PC requires a synchronous "
                    "network model with bounded message delays; under network partitions in asynchronous networks, 3PC can produce split-brain divergent commits.",
                ),
                37: (
                    "4.5 Distributed Two-Phase Locking (2PL) and Deadlock Detection",
                    "Strict Two-Phase Locking (Strict 2PL) guarantees serializability by enforcing two rules: first, a transaction must acquire a shared (S) lock "
                    "prior to reading and an exclusive (X) lock prior to writing; second, all acquired locks must be held continuously until transaction commit or abort, "
                    "preventing dirty reads and cascading aborts. In distributed environments, concurrent transactions executing across multiple shards can form distributed "
                    "deadlock cycles (e.g., T1 holds lock on Node A waiting for Node B, while T2 holds lock on Node B waiting for Node A). Distributed engines detect deadlocks "
                    "using centralized wait-for-graph collectors or edge-chasing probe algorithms.",
                ),
                38: (
                    "4.6 Optimistic Concurrency Control (OCC) and Backward Validation",
                    "Optimistic Concurrency Control (OCC) assumes transaction conflicts are rare in low-contention environments. OCC executes transactions across three "
                    "phases: Read Phase, Validation Phase, and Write Phase. During the Read Phase, transactions read committed values from the database and buffer all updates "
                    "into private local memory workspace without acquiring locks. In the Validation Phase, the engine verifies whether any data items read by the transaction "
                    "were modified by concurrent transactions committed since this transaction's start time. If validation succeeds, changes are flushed to persistent storage "
                    "in the Write Phase; if validation fails, the transaction is immediately aborted and restarted.",
                ),
                39: (
                    "4.7 Multi-Version Concurrency Control (MVCC) and Snapshot Isolation",
                    "Multi-Version Concurrency Control (MVCC) eliminates read-write lock contention by creating an immutable new version of a data row upon every write "
                    "operation. Each row version is annotated with creation timestamp t_create and deletion timestamp t_delete. Under Snapshot Isolation (SI), a transaction T "
                    "assigned start timestamp t_start observes an immutable consistent snapshot containing all row versions where t_create <= t_start and (t_delete is null "
                    "or t_delete > t_start). Readers never block writers, and writers never block readers. Snapshot isolation prevents dirty reads, non-repeatable reads, "
                    "and phantom reads, but permits write skew anomalies under concurrent independent updates.",
                ),
                40: (
                    "4.8 Google Spanner Architecture: TrueTime and External Consistency",
                    "Google Spanner is the first globally distributed database to achieve strict serializability (external consistency) across wide-area multi-datacenter "
                    "clusters. External consistency guarantees that if transaction T2 begins after transaction T1 commits in absolute real-world physical time, T2's commit "
                    "timestamp is strictly greater than T1's commit timestamp (s2 > s1). Spanner achieves this using the TrueTime API, which exposes physical time as an "
                    "interval [earliest, latest] with bounded uncertainty epsilon = (latest - earliest) / 2, typically under 7 milliseconds using GPS receivers and atomic clocks. "
                    "Spanner implements commit wait: a transaction leader delays its commit acknowledgment until TrueTime.now().earliest > s_commit, ensuring uncertainty intervals cannot overlap.",
                ),
                41: (
                    "4.9 Two-Phase Commit Combined with Consensus Leader Leases",
                    "Modern distributed databases eliminate the single point of failure in classical Two-Phase Commit by replacing individual participant nodes "
                    "with Raft or Paxos consensus replica groups. In this architecture (utilized by Google Spanner, CockroachDB, and TiDB), the transaction coordinator "
                    "and each participating shard are themselves fault-tolerant consensus clusters. If a shard leader crashes during the commit phase, the surviving "
                    "Raft followers elect a new leader that reads the replicated in-doubt intent from the distributed Raft log, safely resolving the transaction without blocking.",
                ),
                42: (
                    "4.10 Chapter Summary and Review Questions",
                    "Distributed transactions require balancing consistency, latency, and availability. While classical Two-Phase Commit suffers from coordinator blocking "
                    "hazards, combining 2PC with underlying Paxos/Raft replication and hardware-assisted physical time synchronization (TrueTime) provides globally linearizable "
                    "ACID transactions at planetary scale. System designers must carefully evaluate whether application requirements necessitate full multi-shard ACID "
                    "or can operate under eventual consistency models.",
                ),
            },
        },
        5: {
            "title": "Stream Processing & Asynchronous Event Systems",
            "pages": {
                43: (
                    "5.1 Messaging Paradigms: Message Queues vs Distributed Commit Logs",
                    "Asynchronous event-driven architectures decouple service communication. Traditional message brokers (such as RabbitMQ) employ the destructive "
                    "queue paradigm: messages are published to an exchange, routed to queues, and permanently deleted once consumed and acknowledged by a subscriber. "
                    "In contrast, distributed commit log architectures (such as Apache Kafka and Apache Pulsar) treat streams as immutable, persistent, partitioned append-only "
                    "logs. Messages are retained on disk for a configurable duration (days, months, or indefinitely), allowing multiple independent consumer groups to read "
                    "the same stream at individual processing rates without deleting data.",
                ),
                44: (
                    "5.2 Apache Kafka Partitioning, Offsets, and Consumer Groups",
                    "An Apache Kafka topic is divided into one or more sequential partitions distributed across broker nodes. A partition is an immutable ordered sequence of "
                    "messages, where each message is assigned a monotonically increasing 64-bit integer identifier termed an offset. Producers assign messages to partitions "
                    "using key-based hash partitioning (MurmurHash2(key) mod num_partitions) or round-robin distribution. Within a consumer group, each partition is consumed "
                    "by exactly one consumer instance at any given time, enabling parallel scale-out processing while preserving strict message ordering within each individual partition.",
                ),
                45: (
                    "5.3 Delivery Guarantees: At-Least-Once, At-Most-Once, and Exactly-Once",
                    "Distributed messaging systems provide three distinct delivery semantics. At-most-once delivery commits message offsets before processing; if the consumer "
                    "crashes during processing, the message is permanently lost. At-least-once delivery acknowledges offsets only after message processing completes; if the consumer "
                    "crashes before offset commit, duplicate processing occurs upon restart. Exactly-once semantics (EOS) guarantee that every message influences downstream state "
                    "exactly once. Kafka achieves EOS through idempotent producer sequence numbering and two-phase atomic transaction coordinates spanning input offsets and output topic writes.",
                ),
                46: (
                    "5.4 Event Sourcing and CQRS Architecture Patterns",
                    "Event Sourcing is an architectural pattern where state changes are recorded as an immutable append-only sequence of domain events rather than mutating "
                    "current entity records in place. The current state of any entity is reconstructed by replaying all historical events from genesis. Command Query Responsibility "
                    "Segregation (CQRS) separates the write model (optimized for validating business logic and appending events) from read models (materialized views denormalized "
                    "for high-speed queries). Read models consume the event stream asynchronously, achieving high query performance at the cost of eventual consistency.",
                ),
                47: (
                    "5.5 Stream Processing Time Dimensions: Event Time vs Processing Time",
                    "Stream processing engines (such as Apache Flink and Apache Spark Streaming) distinguish between three distinct temporal domains: Event Time, Ingestion Time, "
                    "and Processing Time. Event Time is the exact physical timestamp when an event occurred at its origin device (embedded in message payload). Ingestion Time is the "
                    "timestamp recorded when the message enters the streaming broker. Processing Time is the local machine clock time of the stream processing operator executing "
                    "the computation. Because network latency and offline mobile clients cause out-of-order and delayed arrivals, robust analytics must be evaluated against Event Time.",
                ),
                48: (
                    "5.6 Watermarks and Out-of-Order Event Handling",
                    "In Event Time stream processing, a Watermark is a specialized control signal embedded into the data stream that establishes a temporal completion threshold. "
                    "A watermark with timestamp W asserts that the system assumes no subsequent events with Event Time t < W will arrive. When an operator observes watermark W, "
                    "it safely triggers and evaluates all tumbling, sliding, or session time windows terminating at or before W. To accommodate real-world network skew, engines "
                    "configure bounded out-of-orderness watermarks: W = max_event_time - delta. Events arriving with timestamp t < W are classified as late data and routed "
                    "to dead-letter side outputs for reprocessing.",
                ),
                49: (
                    "5.7 Windowing Semantics: Tumbling, Sliding, and Session Windows",
                    "Windowing operators group infinite event streams into finite slices for aggregation. Tumbling windows have a fixed duration and do not overlap (e.g., 5-minute "
                    "non-overlapping windows: [00:00, 00:05), [00:05, 00:10)). Sliding windows have a fixed duration and a slide frequency smaller than duration, producing "
                    "overlapping slices (e.g., 5-minute duration with 1-minute slide: [00:00, 00:05), [00:01, 00:06)). Session windows group events into dynamic intervals defined "
                    "by periods of user activity separated by inactivity gap thresholds delta_gap. If no events arrive within delta_gap, the current session window closes.",
                ),
                50: (
                    "5.8 Fault Tolerance in Stream Engines: Chandy-Lamport Distributed Snapshots",
                    "Stream processing engines achieve fault-tolerant stateful computation using distributed checkpointing based on the Chandy-Lamport snapshot algorithm. "
                    "Specialized Checkpoint Barrier markers are injected into input stream sources and flow through operator directed acyclic graphs (DAGs). When an operator "
                    "receives barrier n from all input channels, it freezes processing, snapshots its local operator state (e.g., RocksDB state store) asynchronously to durable "
                    "object storage (such as AWS S3 or HDFS), and forwards the barrier to downstream operators. Upon failure, all operators rollback to checkpoint n, replaying "
                    "source log offsets to resume processing with exact consistency.",
                ),
                51: (
                    "5.9 Backpressure Mechanisms and Reactive Streams",
                    "Backpressure is a flow-control mechanism that prevents fast upstream producers from overwhelming slow downstream consumers, avoiding memory exhaustion and "
                    "Out-Of-Memory (OOM) crashes. In TCP-based streaming pipelines, when a consumer's input buffer fills up, its TCP receive window shrinks to zero, halting sender "
                    "socket transmission. In Apache Flink, backpressure propagates upstream credit-based flow control across network Netty channels, pausing task scheduling "
                    "at source connectors until downstream buffers clear.",
                ),
                52: (
                    "5.10 Chapter Summary and Stream Analytics Review",
                    "Chapter 5 explored event-driven architectures and modern stream analytics. Distributed commit logs (Kafka) provide immutable, replayable event storage. "
                    "Robust stream processing requires decoupling Event Time from Processing Time using watermarks to handle out-of-order data, while distributed barrier "
                    "checkpointing (Chandy-Lamport) guarantees exactly-once stateful processing across arbitrary node failures.",
                ),
            },
        },
        6: {
            "title": "Vector Databases & High-Dimensional Similarity Search",
            "pages": {
                53: (
                    "6.1 High-Dimensional Vector Embeddings and Vector Search",
                    "Modern neural information retrieval and foundation models transform unstructured text, images, and audio into high-dimensional dense vector embeddings "
                    "in R^D (commonly D = 384, 768, 1536, or 3072 dimensions). In this latent semantic space, geometric proximity corresponds to conceptual relatedness. "
                    "Vector databases are purpose-built storage engines optimized for indexing, filtering, and executing k-Nearest Neighbor (k-NN) similarity queries over "
                    "billions of high-dimensional vectors. Executing exact brute-force k-NN requires computing pairwise distances against every vector in the dataset with "
                    "computational complexity O(N * D), which becomes latency-prohibitive for large-scale production applications (exceeding hundreds of milliseconds).",
                ),
                54: (
                    "6.2 Vector Distance Metrics: Euclidean, Cosine, and Inner Product",
                    "Similarity between two high-dimensional vectors u and v is computed using three primary distance metrics: Euclidean (L2) distance, Inner Product (dot product), "
                    "and Cosine distance. Euclidean distance measures the geometric length of the segment connecting two points: L2(u, v) = sqrt(sum((u_i - v_i)^2)). "
                    "Inner Product computes IP(u, v) = sum(u_i * v_i). Cosine similarity measures the angular difference between vectors irrespective of magnitude: "
                    "Cos(u, v) = (u . v) / (||u|| * ||v||). If all vector embeddings are pre-normalized to unit length (||u||_2 = 1.0), Cosine similarity is mathematically "
                    "identical to Inner Product, and Euclidean distance simplifies to L2^2 = 2 * (1 - Cos(u, v)), allowing SIMD vector accelerators to execute ultra-fast dot products.",
                ),
                55: (
                    "6.3 The Curse of Dimensionality in Vector Spaces",
                    "In high-dimensional spaces (D >= 100), geometric intuition from 3D space breaks down due to the Curse of Dimensionality. As dimension D increases, the volume "
                    "of the hypersphere concentrates almost entirely in a thin outer shell near the surface, and the ratio between the distance to the nearest neighbor and the "
                    "distance to the furthest neighbor rapidly approaches 1.0. Traditional spatial partitioning index structures (such as k-d trees and R-trees) degenerate to "
                    "worse than linear scan performance when D > 10. Consequently, scalable vector search relies on Approximate Nearest Neighbor (ANN) index structures.",
                ),
                56: (
                    "6.4 Inverted File Index (IVF) and Voronoi Cells",
                    "The Inverted File Index (IVF) is an Approximate Nearest Neighbor indexing technique that partitions high-dimensional space into Voronoi cells using "
                    "k-means clustering. During index training, k centroid vectors {c_1, c_2, ..., c_k} are learned. Each database vector is assigned to its nearest centroid, "
                    "forming an inverted list of vector IDs for each cluster. At query time, the query vector q is compared against all k centroids to identify the n_probe "
                    "closest centroids (typically n_probe << k). The search scans only the vectors contained within those n_probe inverted lists, reducing search complexity "
                    "from O(N * D) to O(k * D + (n_probe / k) * N * D).",
                ),
                57: (
                    "6.5 Hierarchical Navigable Small World (HNSW) Graphs",
                    "Hierarchical Navigable Small World (HNSW) is the state-of-the-art graph-based ANN index algorithm developed by Malkov and Yashunin. HNSW structures "
                    "vectors into a multi-layer hierarchical graph inspired by skip-lists. The bottom layer (Layer 0) contains all database vectors connected in a dense "
                    "proximity graph. Upper layers contain exponentially fewer vectors (governed by decay parameter m_L = 1 / ln(M)), forming long-range highway links. "
                    "Search begins at the top entry layer with greedy routing, jumping across large distances to locate the local minimum, and descends layer by layer "
                    "until reaching Layer 0, where beam search with candidate size ef_search identifies the top-k nearest neighbors in O(log N) time.",
                ),
                58: (
                    "6.6 HNSW Construction Parameters: M, efConstruction, and efSearch",
                    "HNSW indexing performance is governed by three critical configuration parameters: M, efConstruction, and efSearch. Parameter M specifies the maximum "
                    "number of bidirectional connections (edges) created per node in Layer 0 (typically M = 16 to 64). Parameter efConstruction controls the size of the "
                    "priority queue maintained during index construction; larger efConstruction values improve graph connectivity and recall at the cost of longer build times. "
                    "Parameter efSearch specifies the size of the dynamic candidate list evaluated during query execution; increasing efSearch improves search Recall@K "
                    "with a linear increase in query latency. HNSW achieves over 95% Recall@10 at sub-5 millisecond latencies.",
                ),
                59: (
                    "6.7 Product Quantization (PQ) and Memory Compression",
                    "High-dimensional vector storage in RAM introduces immense memory costs: storing 100 million 768-dimensional float32 vectors requires 307.2 GB of RAM "
                    "exclusive of index graph structures. Product Quantization (PQ) compresses vectors by decomposing high-dimensional space R^D into M orthogonal "
                    "subspaces of dimension d = D / M. In each subspace, k-means clustering identifies K_centroids = 256 sub-centroids. Each sub-vector is replaced "
                    "by the 8-bit (1-byte) index of its nearest centroid. A 768-dimensional vector (3,072 bytes) compressed with M = 96 subspaces requires only 96 bytes, "
                    "achieving a 32x compression ratio. Asymmetric Distance Computation (ADC) enables fast query-to-quantized-code distance lookups using precomputed lookup tables.",
                ),
                60: (
                    "6.8 Hybrid Search: Combining Dense Vectors with Sparse Lexical Inverted Indexes",
                    "While dense vector retrieval excels at capturing semantic conceptual intent, it frequently struggles with exact alphanumeric tokens, product SKU codes, "
                    "proper nouns, and specific acronyms. Hybrid search combines dense vector retrieval (bi-encoder semantic search) with sparse lexical retrieval (BM25 inverted "
                    "index). Results from both dense and sparse retrieval stages are fused using Reciprocal Rank Fusion (RRF): RRF_score(d) = sum(1 / (k + rank_i(d))), where k is a "
                    "smoothing constant (typically k = 60). Alternatively, learned cross-attention models compute convex linear combinations: Score = alpha * Dense + (1 - alpha) * Sparse.",
                ),
                61: (
                    "6.9 Filtered Vector Search: Post-Filtering, Pre-Filtering, and Single-Stage",
                    "Real-world queries require combining vector similarity with relational metadata filters (e.g., status = 'active' and tenant_id = 'org_42'). Post-filtering "
                    "executes standard ANN vector search first and subsequently discards items that fail metadata criteria; if metadata selectivity is high (matching < 1% of data), "
                    "post-filtering returns empty or truncated results. Pre-filtering evaluates metadata filters first to construct a candidate ID set and executes exact k-NN; "
                    "this incurs high latency if millions of items match. Single-stage filtered HNSW modifies graph traversal to evaluate metadata predicates during edge traversal, "
                    "maintaining full top-k recall regardless of filter selectivity.",
                ),
                62: (
                    "6.10 Chapter Summary and Vector Database Architecture",
                    "Vector databases bridge unstructured data representations and neural retrieval. HNSW provides optimal search latency and recall for in-memory graph search, "
                    "while IVF and Product Quantization enable scalable disk-backed and memory-compressed vector storage. Production RAG systems integrate hybrid search and "
                    "single-stage metadata filtering to deliver accurate contextual retrieval for downstream generation models.",
                ),
            },
        },
        7: {
            "title": "Neural Information Retrieval & Cross-Encoder Reranking",
            "pages": {
                63: (
                    "7.1 Evolution of Information Retrieval: Lexical to Neural",
                    "Traditional information retrieval relies on term matching algorithms such as TF-IDF and BM25 (Best Matching 25). BM25 ranks documents based on term "
                    "frequency, inverse document frequency (IDF), and document length normalization: Score(D, Q) = sum(IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)))). "
                    "While BM25 is computationally efficient and robust, it suffers from the vocabulary mismatch problem: queries using synonyms or conceptual paraphrases "
                    "(e.g., 'cardiac arrest' vs 'heart attack') produce zero lexical overlap. Neural information retrieval addresses vocabulary mismatch by learning dense vector "
                    "representations that map semantically equivalent tokens to adjacent coordinates in latent vector space.",
                ),
                64: (
                    "7.2 Bi-Encoder Architectures (Dense Passage Retrieval)",
                    "In a Bi-Encoder architecture (such as DPR or sentence-transformers all-MiniLM-L6-v2), query text and document passages are encoded independently "
                    "into isolated vector embeddings using separate transformer passes: u = E_Q(q) and v = E_D(d). Because document representations are completely independent "
                    "of the query, all documents in the corpus are pre-encoded and indexed offline in an ANN vector database. At query runtime, the query vector is computed in "
                    "a single forward pass, and top-k candidate passages are retrieved via approximate nearest neighbor search in sub-10 milliseconds. However, because query and "
                    "passage tokens never interact through cross-attention, bi-encoders cannot resolve subtle token-level nuances or syntactic modifier relationships.",
                ),
                65: (
                    "7.3 Cross-Encoder Reranking Architecture",
                    "Cross-Encoder models (such as ms-marco-MiniLM-L-6-v2) concatenate the query text and a candidate passage into a single unified input sequence separated by "
                    "the special token: [CLS] Query [SEP] Passage [SEP]. The concatenated sequence passes through all transformer layers, enabling every query token to attend directly "
                    "to every passage token via full bidirectional self-attention across all heads. The final hidden representation of the [CLS] token is projected through a linear "
                    "classification head to produce a scalar relevance score. Cross-encoders provide substantially sharper relevance discrimination than bi-encoders because "
                    "they model complex token interactions, negation, and fine-grained syntactic dependencies.",
                ),
                66: (
                    "7.4 Two-Stage Retrieval Cascade Architecture",
                    "Deploying a Cross-Encoder over an entire corpus of 1 million passages is computationally intractable: evaluating 1 million passages with a 12-layer transformer "
                    "would require thousands of seconds per query. Production RAG systems resolve this latency bottleneck using a Two-Stage Retrieval Cascade. Stage 1 (Fast Retrieval) "
                    "uses a Bi-Encoder with an ANN index (FAISS or pgvector) to retrieve candidate_k = 50 to 100 candidate passages in under 10 milliseconds, maximizing candidate recall. "
                    "Stage 2 (Precision Reranking) applies a Cross-Encoder exclusively to the top candidate_k passages, reranking them with full cross-attention to select the final "
                    "top_k = 3 to 5 passages for LLM context generation. This cascade achieves the precision of full cross-attention within an acceptable 50 to 100 millisecond budget.",
                ),
                67: (
                    "7.5 Computational Complexity and Latency Profiles",
                    "The computational complexity of transformer self-attention is quadratic in sequence length: O(L^2). In a Bi-Encoder, query length L_Q and passage length L_D "
                    "are processed in separate sequences of length ~32 and ~256 tokens respectively, yielding attention operations of O(L_Q^2) + O(L_D^2). In a Cross-Encoder, the sequence "
                    "length is L_total = L_Q + L_D + 3 (~300 tokens), yielding attention complexity of O((L_Q + L_D)^2) = O(L_Q^2 + 2 * L_Q * L_D + L_D^2). Furthermore, while bi-encoder "
                    "document embeddings are computed once offline, cross-encoders must execute online forward passes for every candidate at query time. For candidate_k = 20 on CPU, "
                    "Cross-Encoder reranking typically requires 150 to 300 milliseconds.",
                ),
                68: (
                    "7.6 Information Retrieval Evaluation Metrics: Recall@K and Precision@K",
                    "Information retrieval performance is rigorously evaluated using standard ranking metrics. For a set of ground-truth relevant items Rel and retrieved items Ret: "
                    "Recall@K measures the fraction of relevant items successfully retrieved in the top K positions: Recall@K = |Ret[:K] intersect Rel| / |Rel|. "
                    "Precision@K measures the fraction of the top K retrieved items that are relevant: Precision@K = |Ret[:K] intersect Rel| / K. "
                    "In multi-stage RAG pipelines, Stage 1 is optimized for high Recall@K (ensuring the true answer passage enters the candidate pool), while Stage 2 is optimized "
                    "for high Precision@1 and Precision@3 (ensuring the most relevant passage occupies the top context position).",
                ),
                69: (
                    "7.7 Mean Reciprocal Rank (MRR) and NDCG",
                    "Mean Reciprocal Rank (MRR) evaluates the quality of the first relevant retrieved document across a set of evaluation queries Q: "
                    "MRR = (1 / |Q|) * sum(1 / rank_i), where rank_i is the 1-based rank position of the first relevant document for query i. If no relevant document is retrieved, "
                    "the reciprocal rank is 0. Normalized Discounted Cumulative Gain (NDCG@K) accounts for graded relevance judgments: DCG@K = sum((2^rel_i - 1) / log2(i + 1)), "
                    "and NDCG@K = DCG@K / IDCG@K, where IDCG@K is the ideal DCG achieved by perfect sorting. MRR is especially suited for question answering where a single "
                    "supporting passage suffices to answer the question.",
                ),
                70: (
                    "7.8 Measuring Reranker Rank Movement and Empirical Deltas",
                    "Evaluating the empirical efficacy of a Cross-Encoder reranker requires measuring performance deltas against the baseline Stage 1 vector search: "
                    "Delta MRR = MRR_Stage2 - MRR_Stage1. A positive Delta MRR demonstrates that the reranker successfully promoted relevant passages above irrelevant ones. "
                    "Individual queries are categorized into three rank movement classes: Improved (first relevant item rank decreased, moving closer to rank 1), Unchanged (rank position "
                    "remained identical), and Degraded (relevant item pushed downward by reranker error). If Stage 1 already achieves MRR = 1.0 (first item relevant), Delta MRR is "
                    "identically 0.0, and reranking provides zero measurable ranking improvement.",
                ),
                71: (
                    "7.9 Chunking Strategies and Document Provenance Tracking",
                    "Effective retrieval depends heavily on document chunking strategy. Fixed-size chunking with sliding character or token windows (e.g., target 512 tokens with "
                    "64 token overlap) provides uniform vector representations but risks splitting semantic sentences across boundaries. Sentence-aware and structural chunking "
                    "split text at natural paragraph or heading boundaries, preserving semantic integrity. Every chunk must retain immutable provenance metadata: unique chunk_id, "
                    "document_id, page_number, start_char, and end_char offsets. Provenance metadata enables downstream citation mapping and hallucination auditing.",
                ),
                72: (
                    "7.10 Chapter Summary: Retrieval Cascade Engineering",
                    "Chapter 7 analyzed modern neural information retrieval. Bi-encoders provide fast, scalable approximate nearest neighbor search over precomputed embeddings. "
                    "Cross-encoders deliver high-precision relevance scoring through full cross-attention. Combining both in a two-stage cascade balances sub-100 millisecond "
                    "latency with state-of-the-art ranking precision, evaluated through Recall@K, Precision@K, and Mean Reciprocal Rank.",
                ),
            },
        },
        8: {
            "title": "Large Language Model Serving & KV Cache Optimization",
            "pages": {
                73: (
                    "8.1 Autoregressive Decoder Inference and Token Generation",
                    "Modern generative Large Language Models (such as GPT-4, Llama 3, and FLAN-T5) utilize autoregressive transformer decoders. Given prompt token sequence "
                    "x = [x_1, x_2, ..., x_t], generation proceeds in a sequential loop: the model computes conditional probability distribution P(x_{t+1} | x_1, ..., x_t), "
                    "samples next token x_{t+1}, appends it to sequence x, and repeats until an end-of-sequence token is generated. Inference decomposes into two distinct phases: "
                    "the Prefill Phase (processing all prompt tokens in parallel, which is compute-bound) and the Decode Phase (generating one token at a time sequentially, "
                    "which is memory bandwidth-bound).",
                ),
                74: (
                    "8.2 The Key-Value (KV) Cache Memory Bottleneck",
                    "In self-attention, computing attention for token t requires query vector q_t to multiply key vectors k_1...k_t and value vectors v_1...v_t across all preceding tokens. "
                    "To avoid redundant recomputation of past key and value projections, inference engines cache key and value vectors in GPU High Bandwidth Memory (HBM), termed the KV Cache. "
                    "The memory footprint of the KV cache grows linearly with sequence length: Memory = 2 * b * l * h * d_k * n_bytes, where b is batch size, l is sequence length, "
                    "h is number of attention heads, and d_k is head dimension. For a 13-billion parameter model with 40 layers, 40 heads, and FP16 precision, a single 4096-token "
                    "context consumes over 1.6 GB of GPU RAM exclusively for its KV cache, severely constraining concurrent batch capacity.",
                ),
                75: (
                    "8.3 PagedAttention and vLLM Architecture",
                    "Traditional LLM serving engines pre-allocate contiguous memory chunks for the maximum possible sequence length (e.g., 4096 tokens), resulting in severe memory "
                    "fragmentation (up to 60% to 80% wasted VRAM due to internal, external, and reservation fragmentation). Kwon et al. introduced PagedAttention and the vLLM architecture, "
                    "inspired by virtual memory paging in operating systems. PagedAttention divides the KV cache into fixed-size physical blocks (e.g., 16 or 32 tokens per block). "
                    "The engine maintains a Block Table mapping logical token positions to non-contiguous physical memory pages, eliminating internal fragmentation and enabling near-zero "
                    "memory waste (under 4%). Furthermore, PagedAttention enables zero-copy parallel sampling and prompt sharing.",
                ),
                76: (
                    "8.4 Continuous Batching (Iteration-Level Scheduling)",
                    "Traditional request-level batching waits for all requests in a batch to complete generation before scheduling new requests. Because request sequence lengths vary "
                    "widely, early-completing requests remain idle while waiting for the longest request to finish, resulting in severe GPU compute underutilization. Continuous batching "
                    "(also termed iteration-level scheduling, implemented in Orca and vLLM) operates at the granularity of individual token generation iterations. As soon as a request "
                    "emits an EOS token, its memory pages are evicted, and a new incoming request is immediately inserted into the active batch without stalling existing requests.",
                ),
                77: (
                    "8.5 Model Parallelism: Tensor Parallelism vs Pipeline Parallelism",
                    "When model weights exceed single GPU VRAM capacity, model parallelism partitions model execution across multiple accelerators. Tensor Parallelism (TP, Megatron-LM) "
                    "shards individual weight matrices across GPUs within the same server node connected via high-speed NVLink (e.g., column-parallel projection for self-attention W_Q, W_K, W_V "
                    "and row-parallel projection for W_O), requiring all-reduce synchronization operations per transformer layer. Pipeline Parallelism (PP) partitions consecutive layers "
                    "across distinct nodes connected via InfiniBand, pipelining micro-batches to minimize idle pipeline bubble time.",
                ),
                78: (
                    "8.6 Quantization: FP16, INT8, and INT4 Weight Formats",
                    "Model quantization reduces memory bandwidth pressure and storage requirements by compressing weight representations. FP16 and BF16 use 16 bits (2 bytes) per parameter. "
                    "Post-Training Quantization (PTQ) techniques (such as AWQ and GPTQ) compress weights to INT4 (4 bits, 0.5 bytes per parameter) with negligible perplexity degradation. "
                    "Quantizing a 70-billion parameter model reduces weight memory from 140 GB (FP16) down to ~35 GB (INT4), allowing the model to fit on a single consumer GPU "
                    "or dramatically increasing batch throughput on server hardware. In RAG serving, smaller encoder and reader models (like FLAN-T5-base at 250M parameters) execute "
                    "efficiently on standard CPU hardware.",
                ),
                79: (
                    "8.7 Speculative Decoding and Draft Models",
                    "Autoregressive decoding is bottlenecked by sequential memory reads from HBM to processor registers for each generated token. Speculative decoding accelerates "
                    "inference using a small, high-speed draft model (e.g., 1B parameter model) to speculatively generate k candidate tokens in sequence. The primary large model "
                    "(e.g., 70B parameter model) evaluates all k candidate tokens in parallel in a single forward pass using modified causal attention masks. Because checking k tokens "
                    "in parallel takes approximately the same time as generating one token, accepting m <= k tokens yields a 2x to 3x inference speedup with mathematically identical output.",
                ),
                80: (
                    "8.8 Extractive Question Answering vs Generative Synthesis",
                    "In RAG architectures, reader models are categorized into extractive and generative paradigms. Extractive QA models (such as RoBERTa fine-tuned on SQuAD 2.0) "
                    "predict start and end token indices directly from the retrieved context passage, extracting verbatim spans. Extractive QA guarantees zero hallucination "
                    "because all answers are literal quotes from the source text; however, it cannot synthesize multi-sentence explanations or format complex outputs. Generative "
                    "models (such as FLAN-T5 and modern decoder LLMs) synthesize abstractive, fluent natural language answers, requiring external NLI validation to prevent hallucination.",
                ),
                81: (
                    "8.9 Serving Latency Components: TTFT and TPOT",
                    "Production LLM serving performance is measured by two primary latency metrics: Time To First Token (TTFT) and Time Per Output Token (TPOT). "
                    "TTFT measures the time elapsed from user request transmission until the first generated token arrives, representing prefill computation latency and queuing delay. "
                    "TPOT (also termed inter-token latency) measures the time required to generate each subsequent token during the decoding phase. In interactive conversational "
                    "applications, human perception requires TTFT under 500 milliseconds and TPOT under 50 milliseconds per token (20 tokens per second).",
                ),
                82: (
                    "8.10 Chapter Summary and Serving Optimization Principles",
                    "Chapter 8 analyzed LLM serving systems and inference optimization. The KV cache represents the primary memory bottleneck in autoregressive generation. "
                    "PagedAttention eliminates memory fragmentation, while continuous batching maximizes GPU utilization. Serving architectures must optimize TTFT and TPOT "
                    "while selecting appropriate model architectures (extractive vs generative) based on grounding and synthesis requirements.",
                ),
            },
        },
        9: {
            "title": "Distributed Caching & Memory Consistency",
            "pages": {
                83: (
                    "9.1 The Role of Distributed Caching in Scalable Systems",
                    "In modern web and enterprise architectures, distributed caching stores high-frequency, low-latency transient data in volatile RAM (such as Redis or Memcached). "
                    "By serving read requests directly from memory, caches reduce database query load by 80% to 95% and lower read latency from tens of milliseconds to sub-millisecond "
                    "scales. However, caching introduces distributed state synchronization challenges, cache coherence trade-offs, and stale data risks under concurrent updates.",
                ),
                84: (
                    "9.2 Cache Topologies: Cache-Aside, Read-Through, and Write-Through",
                    "Distributed caching employs three primary operational patterns: Cache-Aside (Lazy Loading), Read-Through, and Write-Through. In the Cache-Aside pattern, "
                    "application code queries the cache first; on a cache miss, the application queries the database, writes the result to the cache, and returns it to the client. "
                    "In Read-Through caching, the application interacts exclusively with the cache, and the cache layer automatically queries the datastore on a miss. "
                    "In Write-Through caching, writes update the cache and database synchronously within a unified transaction, guaranteeing cache freshness at the cost of higher write latency.",
                ),
                85: (
                    "9.3 Write-Behind (Write-Back) Caching and Durability Hazards",
                    "Write-Behind (Write-Back) caching acknowledges write requests immediately after updating the in-memory cache and enqueues modified data records into an "
                    "asynchronous batch buffer for background flushing to the database. Write-Behind provides ultra-fast write performance and absorbs sudden traffic spikes; "
                    "however, if cache nodes crash or experience power loss before dirty buffers flush to persistent storage, data modifications are permanently lost. "
                    "Write-behind is utilized in non-critical metrics counters and real-time gaming session state.",
                ),
                86: (
                    "9.4 Cache Eviction Algorithms: LRU, LFU, and ARC",
                    "Because RAM capacity is finite, caches employ eviction algorithms to remove items when memory limits are reached. Least Recently Used (LRU) evicts the item "
                    "that has not been accessed for the longest period, implemented using a doubly-linked list combined with a hash map in O(1) time. Least Frequently Used (LFU) "
                    "tracks access counters and evicts items with the lowest cumulative request frequency. Adaptive Replacement Cache (ARC) dynamically balances between recency "
                    "and frequency by maintaining two separate LRU lists for recent and frequent pages, outperforming static LRU under changing access patterns.",
                ),
                87: (
                    "9.5 The Cache Stampede Problem and Probabilistic Early Expiration",
                    "A Cache Stampede (also known as a Thundering Herd problem) occurs when a popular cached key expires under high concurrent load. Thousands of simultaneous "
                    "client requests experience a cache miss at the exact same millisecond and concurrently issue expensive queries to the backend database, causing database "
                    "CPU spikes, connection exhaustion, and catastrophic service failure. Cache stampedes are mitigated using distributed mutex locks or the XFetch probabilistic "
                    "early expiration algorithm: early_recompute = -beta * delta * ln(random()), which asynchronously recomputes and refreshes the cache before actual TTL expiry.",
                ),
                88: (
                    "9.6 Cache Invalidation: Dual-Write Inconsistencies and TTLs",
                    "Phil Karlton famously observed that cache invalidation is one of the two hardest problems in computer science. When updating an entity, performing dual writes "
                    "(updating the database and updating the cache) creates race conditions under concurrent requests (e.g., T1 writes DB, T2 writes DB, T2 writes Cache, T1 writes Cache, "
                    "leaving the cache permanently inconsistent with the database). To maintain consistency, applications should delete the cache key rather than updating it, "
                    "or rely on Change Data Capture (CDC) streams from database WAL logs to invalidate cache entries asynchronously.",
                ),
                89: (
                    "9.7 Distributed Locking: The Redlock Algorithm",
                    "Distributed systems require distributed mutual exclusion locks to coordinate non-idempotent operations across independent workers. The Redlock algorithm "
                    "(proposed by Sanfilippo for Redis) establishes fault-tolerant distributed locking across N independent Redis master nodes (typically N = 5). A client acquires "
                    "the lock by attempting to set a key with a unique random value and small timeout on all N instances sequentially. The lock is successfully acquired if the client "
                    "secures keys on a majority (floor(N / 2) + 1 = 3) of instances within a validity duration strictly less than the lock TTL.",
                ),
                90: (
                    "9.8 Redis Cluster Sharding and Sentinel Failover",
                    "Redis Cluster scales memory capacity horizontally across nodes using 16,384 fixed Hash Slots. Keys are assigned to hash slots using CRC16: slot = CRC16(key) mod 16384. "
                    "Each master node is responsible for a subset of the 16,384 slots. High availability is managed by Redis Sentinel or integrated cluster gossip protocols, which monitor "
                    "master health, execute automated leader failover upon crash detection, and promote read replicas to master status without manual operator intervention.",
                ),
                91: (
                    "9.9 Memory Overhead, Serialization, and Compaction in Caching",
                    "In in-memory key-value stores, memory overhead extends beyond raw data bytes to include hash table pointers, jemalloc memory allocation padding, and metadata overhead. "
                    "Storing millions of small string keys can consume 10x more RAM than the underlying payload. High-performance systems compress data using compact binary serialization "
                    "formats (such as Protocol Buffers or MessagePack) rather than JSON, reducing memory footprint by 40% to 60% and cutting serialization CPU overhead.",
                ),
                92: (
                    "9.10 Chapter Summary and Caching Best Practices",
                    "Chapter 9 analyzed distributed caching architectures. Caching accelerates read performance but requires rigorous strategies for invalidation, eviction, and "
                    "concurrency control. Cache stampedes must be mitigated using distributed locking or probabilistic early recomputation, while distributed locking algorithms "
                    "like Redlock provide mutual exclusion across independent nodes.",
                ),
            },
        },
        10: {
            "title": "Site Reliability Engineering, Observability & Security",
            "pages": {
                93: (
                    "10.1 Site Reliability Engineering (SRE) Principles and Error Budgets",
                    "Site Reliability Engineering (SRE), pioneered by Google, applies software engineering disciplines to infrastructure operations. SRE balances service reliability "
                    "against feature development velocity through the concept of Error Budgets. A service level objective (SLO) defines the target reliability percentage (e.g., 99.9% availability, "
                    "or 'three nines'). The error budget is the allowable downtime: Error Budget = 100% - SLO (0.1% downtime, corresponding to 43.8 minutes per month). If a service "
                    "exhausts its error budget due to outages, feature releases are automatically frozen until engineering addresses reliability debt.",
                ),
                94: (
                    "10.2 Service Level Indicators (SLI), Objectives (SLO), and Agreements (SLA)",
                    "Reliability engineering defines three tiered service metrics: Service Level Indicators (SLI), Service Level Objectives (SLO), and Service Level Agreements (SLA). "
                    "An SLI is a quantifiable metric measuring service performance in real time (e.g., fraction of successful HTTP requests: successful_requests / total_requests). "
                    "An SLO is a target reliability threshold agreed upon by internal engineering teams (e.g., SLI >= 99.95% over a 30-day rolling window). An SLA is a formal legal contract "
                    "with external customers that specifies financial penalties or billing credits if the service fails to maintain committed SLO thresholds.",
                ),
                95: (
                    "10.3 The Four Golden Signals of Distributed Observability",
                    "Google's SRE framework identifies the Four Golden Signals for monitoring distributed systems: Latency, Traffic, Errors, and Saturation. Latency measures the time "
                    "required to service a request, distinguishing successful request latency from error latency. Traffic measures the demand placed on the system (e.g., HTTP requests "
                    "per second, network I/O throughput). Errors measures the rate of requests that fail explicitly (HTTP 500) or semantically (wrong content). Saturation measures the "
                    "fraction of system resources currently consumed (e.g., CPU utilization, memory pressure, database connection pool exhaustion).",
                ),
                96: (
                    "10.4 Resilience Patterns: Circuit Breakers, Bulkheads, and Jitter",
                    "Distributed systems must prevent localized failures from cascading into catastrophic cluster collapse. The Circuit Breaker pattern monitors downstream service calls: "
                    "in the Closed state, calls pass through normally; if error rates exceed a threshold, the breaker transitions to Open state, failing fast immediately without calling the "
                    "downstream dependency. After a sleep window, it enters Half-Open state to test recovery. Bulkheads isolate resources into independent pools (e.g., separate thread pools "
                    "per client). Exponential backoff with randomized jitter prevents synchronized retry storms by introducing random delay: t = min(t_max, t_base * 2^attempt) + uniform(0, jitter).",
                ),
                97: (
                    "10.5 Distributed Tracing and OpenTelemetry Context Propagation",
                    "In microservice architectures, a single user request can trigger hundreds of RPC calls across dozens of services. Distributed tracing tracks request execution paths "
                    "using unique Trace IDs and Span IDs. The W3C Trace Context standard defines traceparent headers propagated across HTTP and gRPC network boundaries: "
                    "version-trace_id-parent_id-trace_flags. OpenTelemetry provides vendor-neutral instrumentation APIs to capture execution spans, timing breakdowns, and diagnostic events, "
                    "allowing engineers to pinpoint high-latency bottlenecks and error origins across complex distributed dependency graphs.",
                ),
                98: (
                    "10.6 Zero Trust Architecture and Mutual TLS (mTLS)",
                    "Traditional security models relied on perimeter security ('castle-and-moat'), assuming all traffic inside the corporate datacenter network was trusted. "
                    "Zero Trust Architecture eliminates perimeter trust, enforcing the core principle: 'never trust, always verify'. Every request between microservices must be "
                    "authenticated, authorized, and encrypted. Mutual TLS (mTLS) uses X.509 digital certificates to establish encrypted TLS tunnels where both the client and server "
                    "authenticate each other's identity, preventing man-in-the-middle attacks and eavesdropping across internal datacenter networks.",
                ),
                99: (
                    "10.7 Role-Based and Attribute-Based Access Control (RBAC vs ABAC)",
                    "Authorization models regulate access to protected distributed resources. Role-Based Access Control (RBAC) assigns permissions to predefined organizational roles "
                    "(e.g., Admin, Editor, Viewer), and binds users to roles. While RBAC is simple to administer, it suffers from role explosion in complex multi-tenant environments. "
                    "Attribute-Based Access Control (ABAC) evaluates dynamic access policies based on attributes of the subject (user), action, resource, and environment (e.g., time, location). "
                    "Open Policy Agent (OPA) executes decoupled policy evaluation using the declarative Rego language across cloud-native architectures.",
                ),
                100: (
                    "10.8 Data Protection at Rest and Envelope Encryption",
                    "Securing persistent storage requires encryption at rest using Envelope Encryption. Data is encrypted using a unique symmetric Data Encryption Key (DEK, typically AES-256-GCM). "
                    "The DEK itself is never stored in plaintext on disk; instead, it is encrypted (wrapped) using a Key Encryption Key (KEK) managed by a Hardware Security Module (HSM) "
                    "or cloud Key Management Service (KMS, such as AWS KMS or Google Cloud KMS). To read data, the service requests KMS to decrypt the DEK in volatile memory, avoiding "
                    "transmitting large data payloads over the network to KMS.",
                ),
                101: (
                    "10.9 Chaos Engineering and Fault Injection Testing",
                    "Chaos Engineering is the discipline of experimenting on a distributed system to build confidence in its capability to withstand turbulent conditions in production. "
                    "Pioneered by Netflix's Chaos Monkey, chaos engineering systematically injects simulated faults: terminating random server instances, introducing artificial network "
                    "latency, simulating packet loss, corrupting DNS responses, and triggering disk full conditions. By actively verifying steady-state behavior under injected failures, "
                    "engineering teams discover hidden cascading bugs before they impact production users.",
                ),
                102: (
                    "10.10 Disaster Recovery: RPO, RTO, and Multi-Region Architectures",
                    "Disaster recovery planning evaluates two fundamental recovery objectives: Recovery Point Objective (RPO) and Recovery Time Objective (RTO). RPO measures the maximum "
                    "acceptable duration of data loss following a disaster (e.g., RPO = 5 minutes implies at most 5 minutes of committed data may be lost). RTO measures the maximum "
                    "acceptable duration of service downtime before operational restoration (e.g., RTO = 1 hour). Multi-region active-active architectures achieve near-zero RPO and RTO "
                    "by routing client traffic across geographically dispersed datacenters with global consensus and data replication.",
                ),
                103: (
                    "10.11 Security Auditing, Provenance, and Immutability",
                    "Enterprise regulatory compliance (SOC 2, ISO 27001, HIPAA) mandates comprehensive, tamper-evident audit logging. Audit logs record every administrative action, "
                    "authentication event, and data access query with cryptographic timestamps. Audit logs are written to write-once-read-many (WORM) storage buckets with object lock "
                    "retention policies, preventing malicious or accidental deletion even by privileged administrative credentials. Cryptographic hash chains (Merkle trees) guarantee "
                    "log integrity by making unauthorized modification computationally impossible without invalidating subsequent tree root hashes.",
                ),
                104: (
                    "10.12 Artificial Intelligence System Safety and Evaluation Standards",
                    "Deploying generative AI and RAG architectures in production requires comprehensive safety, grounding, and evaluation protocols. AI systems must guard against "
                    "prompt injection attacks, data poisoning, and ungrounded hallucinations. Reliable RAG systems enforce strict grounding validation using Natural Language Inference (NLI) "
                    "models, safe refusal policies on ambiguous or unanswerable queries, and fine-grained citation provenance tracking. Continuous automated evaluation against "
                    "human-verified golden benchmarks ensures that architectural modifications do not regress retrieval ranking or answer correctness.",
                ),
                105: (
                    "10.13 Epilogue: The Future of Cloud-Native Infrastructure",
                    "The convergence of distributed consensus, high-dimensional vector search, and foundation generative models represents the modern frontier of software engineering. "
                    "Building resilient, reliable, and grounded AI systems demands rigorous adherence to distributed systems foundations: embracing partial synchrony, respecting "
                    "impossibility theorems, optimizing multi-stage retrieval cascades, and relentlessly auditing system quality across comprehensive benchmark suites.",
                ),
            },
        },
    }

    # Generate pages:
    # Page 1: Title page
    p1 = doc.new_page(width=612, height=792)  # Standard Letter
    p1.insert_text(
        fitz.Point(72, 180),
        "Foundations of Distributed Systems,\nCloud Architecture, and Machine\nLearning Infrastructure",
        fontsize=24,
        fontname="helv",
        color=(0.1, 0.2, 0.4),
    )
    p1.insert_text(
        fitz.Point(72, 280),
        "A Comprehensive Engineering Handbook (First Edition)",
        fontsize=14,
        fontname="helv",
        color=(0.3, 0.3, 0.3),
    )
    p1.insert_text(
        fitz.Point(72, 340),
        "Dr. Alexander Vance & Dr. Elena Rostova\nInstitute for Distributed Information Systems",
        fontsize=12,
        fontname="helv",
        color=(0.2, 0.2, 0.2),
    )
    p1.insert_text(
        fitz.Point(72, 700),
        "Document ID: doc_dist_sys_handbook_01 | Published for Production RAG Benchmarking",
        fontsize=9,
        fontname="helv",
        color=(0.5, 0.5, 0.5),
    )

    # Page 2: Table of Contents & Preface
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(
        fitz.Point(72, 72),
        "Table of Contents & Handbook Overview",
        fontsize=18,
        fontname="helv",
        color=(0.1, 0.2, 0.4),
    )
    toc_lines = [
        "Chapter 1: Distributed Systems Foundations & Logical Clocks (Pages 3-12)",
        "Chapter 2: Consensus Protocols & Fault Tolerance: Paxos & Raft (Pages 13-22)",
        "Chapter 3: Storage Architectures, LSM-Trees & Data Partitioning (Pages 23-32)",
        "Chapter 4: Distributed Transactions & Concurrency Control: 2PC & MVCC (Pages 33-42)",
        "Chapter 5: Stream Processing & Asynchronous Event Systems: Kafka & Flink (Pages 43-52)",
        "Chapter 6: Vector Databases & High-Dimensional Similarity Search: HNSW & PQ (Pages 53-62)",
        "Chapter 7: Neural Information Retrieval & Cross-Encoder Reranking (Pages 63-72)",
        "Chapter 8: Large Language Model Serving & KV Cache Optimization: vLLM (Pages 73-82)",
        "Chapter 9: Distributed Caching & Memory Consistency: Redis & Redlock (Pages 83-92)",
        "Chapter 10: Site Reliability Engineering, Observability & Security (Pages 93-105)",
    ]
    y = 120
    for line in toc_lines:
        p2.insert_text(fitz.Point(72, y), line, fontsize=11, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 24

    p2.insert_textbox(
        fitz.Rect(72, y + 20, 540, 720),
        "Preface: This handbook presents rigorous mathematical definitions, architectural analyses, "
        "and empirical trade-offs across modern distributed systems, vector retrieval, and machine learning "
        "inference infrastructure. Each chapter covers theoretical foundations, concrete implementation details, "
        "and operational metrics essential for production systems engineering.",
        fontsize=10,
        fontname="helv",
    )

    # Pages 3 to 105: Chapters 1 to 10
    for ch_num, ch_info in sorted(chapter_data.items()):
        ch_title = ch_info["title"]
        for page_num, (sec_title, body_text) in sorted(ch_info["pages"].items()):
            page = doc.new_page(width=612, height=792)

            # Header
            page.insert_text(
                fitz.Point(72, 45),
                f"Chapter {ch_num}: {ch_title}",
                fontsize=8,
                fontname="helv",
                color=(0.5, 0.5, 0.5),
            )
            page.insert_text(
                fitz.Point(520, 45),
                f"Page {page_num}",
                fontsize=8,
                fontname="helv",
                color=(0.5, 0.5, 0.5),
            )

            # Section Title
            page.insert_text(
                fitz.Point(72, 85),
                sec_title,
                fontsize=14,
                fontname="helv",
                color=(0.1, 0.2, 0.4),
            )

            # Body text in bounded rect
            rect = fitz.Rect(72, 110, 540, 720)
            page.insert_textbox(
                rect,
                body_text,
                fontsize=10.5,
                fontname="helv",
                color=(0.15, 0.15, 0.15),
            )

            # Footer
            page.insert_text(
                fitz.Point(72, 755),
                f"Foundations of Distributed Systems | Document ID: doc_dist_sys_handbook_01",
                fontsize=7.5,
                fontname="helv",
                color=(0.6, 0.6, 0.6),
            )

    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent.parent / "data" / "distributed_systems_handbook.pdf"
    created = create_full_book_pdf(out)
    print(f"Successfully generated {created} with 105 pages!")
