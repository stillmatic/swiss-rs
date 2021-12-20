use mwmatching::Matching;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::Beta;
use std::collections::HashMap;

#[derive(Copy, Clone, Default)]
struct Player {
    id: usize,
    strength: f64,
}

impl Player {
    pub fn new(id: usize, strength: f64) -> Player {
        return Player { id, strength };
    }
}

impl std::hash::Hash for Player {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        state.write_i32(self.id.try_into().unwrap());
        state.finish();
    }
}

impl Eq for Player {}

impl PartialEq for Player {
    fn eq(&self, other: &Player) -> bool {
        self.id == other.id
    }
}

struct Match {
    a: Player,
    b: Player,
}

impl Match {
    fn simulate(&self) -> Player {
        let mut rng = rand::thread_rng();
        let roll: f64 = rng.gen::<f64>();
        let prob_a_wins = self.a.strength / (self.a.strength + self.b.strength);
        if prob_a_wins > roll {
            return self.a;
        } else {
            return self.b;
        };
    }
}

fn generate_players(n_players: usize) -> Vec<Player> {
    // empirically assigned (but mostly arbitrary)
    // TODO: pass in from struct definition or something
    let strength_distribution = Beta::new(2.0, 5.0).unwrap();
    let mut rng = thread_rng();
    let mut v: Vec<Player> = Vec::with_capacity(n_players);
    for i in 0..n_players {
        v.push(Player::new(i, strength_distribution.sample(&mut rng)));
    }
    v
}

struct Simulation {
    players: Vec<Player>,
    n_rounds: usize,
    matches: Graph<Player, i32>,
    nodes: Vec<NodeIndex>,
}

fn run_first_round(
    players: Vec<Player>,
    mut matches: Graph<Player, i32>,
    nodes: Vec<NodeIndex>,
) -> (HashMap<Player, usize>, Graph<Player, i32>) {
    let mut rng = thread_rng();

    // setup stats runner
    let mut results: HashMap<Player, usize> = HashMap::new();
    for player in players.iter() {
        results.insert(*player, 0);
    }

    // setup Graph
    for (l, r) in players.clone().iter().zip(players.clone().iter()) {
        if l.id > r.id {
            matches.update_edge(nodes[l.id], nodes[r.id], 0);
        }
    }

    // find random pairing and run
    let mut opponents: Vec<Player> = players.clone();
    opponents.shuffle(&mut rng);
    // TODO: abstract this out
    for chunk in opponents.chunks(2) {
        let a = chunk[0];
        let b = chunk[1];
        let pairing = Match { a, b };
        let winner = pairing.simulate();
        results.insert(winner, 1 + results[&winner]);
        // super low weight since the round is done
        matches.update_edge(nodes[a.id], nodes[b.id], 10000);
        println! {"{a} ({a_strength:.2}) won vs {b} ({b_strength:.2}) for a total of {n} wins",
        a=a.id, a_strength=a.strength, b_strength=b.strength,b=b.id, n=results[&winner]};
    }
    (results, matches)
}

fn run_round(
    players: Vec<Player>,
    mut matches: Graph<Player, i32>,
    mut results: HashMap<Player, usize>,
    nodes: Vec<NodeIndex>,
) -> (HashMap<Player, usize>, Graph<Player, i32>) {
    // populate edgelist for mwmatching
    let mut edges: Vec<mwmatching::Edge> = Vec::new();
    for edge in matches.edge_references() {
        let source_idx = edge.source().index();
        let target_idx = edge.target().index();
        let mut weight = *edge.weight();
        // if the team hasn't played, update the weights
        let wins_a = results[&players[source_idx]];
        let wins_b = results[&players[target_idx]];
        let diff: i32 = wins_a as i32 - wins_b as i32;
        if weight < 0 {
            if diff.abs() > 1 {
                weight = -2 - diff
            } else {
                weight = -3500 + (35 * diff).pow(2);
            }
        }
        // println! {"{a}: ({wins_a}) - {b}: ({wins_b}) - {weight}", a=source_idx, b=target_idx,wins_a=wins_a, wins_b=wins_b, weight=weight}

        edges.push((source_idx, target_idx, weight));
    }
    let mut pairing = Matching::new(edges);
    let verts = pairing.solve();
    // Returns a list "mates", such that mates[i] == j if vertex i is matched to vertex j,
    // and mates[i] == SENTINEL if vertex i is not matched, where SENTINEL is usize::max_value().`
    for (idx, vert) in verts.iter().enumerate() {
        if idx > *vert {
            let a = players[idx];
            let b = players[*vert];
            let diff: i32 = (results[&a] as i32 - results[&b] as i32).abs();
            // println! {"{a_n} vs {b_n} wins ---- {diff}", a_n=results[&a], b_n=results[&b], diff=diff};
            let pairing = Match { a, b };
            let winner = pairing.simulate();
            results.insert(winner, 1 + results[&winner]);
            // super high weight since the round is done
            matches.update_edge(nodes[idx], nodes[*vert], 10000);
            // println! {"{a} ({a_strength:.2}) won vs {b} ({b_strength:.2}) for a total of {n} wins",
            // a=a.id, a_strength=a.strength, b_strength=b.strength,b=b.id, n=results[&winner]};
        }
    }
    return (results, matches);
}

impl Simulation {
    pub fn new(n_players: usize, n_rounds: usize) -> Simulation {
        let players = generate_players(n_players);
        let mut matches = Graph::<Player, i32>::new();
        let nodes: Vec<NodeIndex> = players
            .clone()
            .into_iter()
            .map(|x| matches.add_node(x))
            .collect();
        Simulation {
            players,
            n_rounds,
            matches,
            nodes,
        }
    }

    pub fn run(mut self) -> () {
        let n_rounds = &self.n_rounds;

        // setup stats runner
        let mut results: HashMap<Player, usize> = HashMap::new();
        for player in self.players.iter() {
            results.insert(*player, 0);
        }

        // setup Graph
        for l in self.players.clone().iter() {
            for r in self.players.clone().iter() {
                if l.id > r.id {
                    self.matches
                        .update_edge(self.nodes[l.id], self.nodes[r.id], 0);
                }
            }
        }

        // iterate
        for _round_num in 0..*n_rounds {
            // println! {"------- this is round {round_num} -------", i=i+1};
            let res = run_round(
                self.players.to_vec(),
                self.matches,
                results,
                self.nodes.to_vec(),
            );
            results = res.0;
            self.matches = res.1;
        }
        let mut res_vec: Vec<(&Player, &usize)> = results.iter().collect();
        res_vec.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        println! {"-------"}
        for res in res_vec {
            println! {"Player {player} ({strength:.2}) won {n} games", player=res.0.id, n=res.1, strength=res.0.strength}
        }
    }
}

fn main() {
    let sim = Simulation::new(64, 6);
    sim.run();
}
