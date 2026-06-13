/// Reasoning task scheduler.
/// Manages deterministic parallel execution of reasoning operations.

use crate::runtime::parallel::DeterministicRuntime;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct ScheduledTask<T> {
    pub id: u64,
    pub payload: T,
}

#[derive(Debug, Default)]
pub struct TaskScheduler {
    runtime: DeterministicRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkStealingPlan {
    pub worker_count: usize,
    pub execution_order: Vec<u64>,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            runtime: DeterministicRuntime::new(),
        }
    }

    pub fn run_deterministic<T, R, F>(&self, mut tasks: Vec<ScheduledTask<T>>, op: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&ScheduledTask<T>) -> R + Send + Sync,
    {
        tasks.sort_by_key(|t| t.id);
        self.runtime.execute_indexed(&tasks, op)
    }

    pub fn build_work_stealing_plan<T>(&self, tasks: &[ScheduledTask<T>], worker_count: usize) -> WorkStealingPlan {
        let workers = worker_count.max(1);
        let mut sorted_ids: Vec<u64> = tasks.iter().map(|t| t.id).collect();
        sorted_ids.sort_unstable();

        let mut queues: Vec<VecDeque<u64>> = (0..workers).map(|_| VecDeque::new()).collect();
        for (idx, task_id) in sorted_ids.iter().enumerate() {
            queues[idx % workers].push_back(*task_id);
        }

        let mut execution_order = Vec::with_capacity(sorted_ids.len());
        while execution_order.len() < sorted_ids.len() {
            for w in 0..workers {
                if let Some(task_id) = queues[w].pop_front() {
                    execution_order.push(task_id);
                    continue;
                }

                let donor = queues
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| q.len() > 1)
                    .max_by_key(|(idx, q)| (q.len(), usize::MAX - *idx))
                    .map(|(idx, _)| idx);

                if let Some(donor_idx) = donor {
                    if let Some(stolen) = queues[donor_idx].pop_back() {
                        execution_order.push(stolen);
                    }
                }
            }
        }

        WorkStealingPlan {
            worker_count: workers,
            execution_order,
        }
    }

    pub fn run_work_stealing_deterministic<T, R, F>(
        &self,
        tasks: Vec<ScheduledTask<T>>,
        worker_count: usize,
        op: F,
    ) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&ScheduledTask<T>) -> R + Send + Sync,
    {
        let plan = self.build_work_stealing_plan(&tasks, worker_count);
        let mut indexed: Vec<(u64, &ScheduledTask<T>)> = tasks.iter().map(|t| (t.id, t)).collect();
        indexed.sort_by_key(|(id, _)| *id);

        let by_id: std::collections::BTreeMap<u64, &ScheduledTask<T>> = indexed.into_iter().collect();
        let ordered_tasks: Vec<&ScheduledTask<T>> = plan
            .execution_order
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .collect();

        self.runtime.execute_indexed(&ordered_tasks, |task| op(task))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_enforces_stable_order() {
        let scheduler = TaskScheduler::new();
        let tasks = vec![
            ScheduledTask { id: 20, payload: 2 },
            ScheduledTask { id: 10, payload: 1 },
        ];

        let out = scheduler.run_deterministic(tasks, |task| task.id);
        assert_eq!(out, vec![10, 20]);
    }

    #[test]
    fn work_stealing_plan_is_reproducible() {
        let scheduler = TaskScheduler::new();
        let tasks = vec![
            ScheduledTask { id: 1, payload: 11 },
            ScheduledTask { id: 2, payload: 12 },
            ScheduledTask { id: 3, payload: 13 },
            ScheduledTask { id: 4, payload: 14 },
            ScheduledTask { id: 5, payload: 15 },
            ScheduledTask { id: 6, payload: 16 },
        ];

        let p1 = scheduler.build_work_stealing_plan(&tasks, 3);
        let p2 = scheduler.build_work_stealing_plan(&tasks, 3);
        assert_eq!(p1, p2);
    }

    #[test]
    fn work_stealing_execution_is_reproducible() {
        let scheduler = TaskScheduler::new();
        let tasks = vec![
            ScheduledTask { id: 30, payload: 3 },
            ScheduledTask { id: 10, payload: 1 },
            ScheduledTask { id: 20, payload: 2 },
            ScheduledTask { id: 40, payload: 4 },
            ScheduledTask { id: 50, payload: 5 },
        ];

        let a = scheduler.run_work_stealing_deterministic(tasks.clone(), 2, |t| t.id * 10);
        let b = scheduler.run_work_stealing_deterministic(tasks, 2, |t| t.id * 10);
        assert_eq!(a, b);
    }
}
