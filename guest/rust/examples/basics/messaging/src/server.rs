use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use ambient_api::prelude::*;

#[main]
pub fn main() {
    messages::Hello::subscribe(|source, data| {
        let Some(user_id) = source.client_user_id() else { return; };
        println!("{user_id}: {:?}", data);

        let source_reliable = data.source_reliable;

        messages::Hello::new(
            true,
            format!("{source_reliable}: Hello, world from the server!"),
        )
        .send_client_targeted_reliable(user_id.clone());

        messages::Hello::new(
            false,
            format!("{source_reliable}: Hello, world from the server!"),
        )
        .send_client_targeted_unreliable(user_id);

        messages::Hello::new(
            true,
            format!("{source_reliable}: Hello, world (everyone) from the server!"),
        )
        .send_client_broadcast_reliable();
    });

    let handled = Arc::new(AtomicBool::new(false));
    messages::Local::subscribe({
        let handled = handled.clone();
        move |source, data| {
            handled.store(true, Ordering::SeqCst);
            println!("{source:?}: {data:?}");
        }
    });
    run_async(async move {
        while !handled.load(Ordering::SeqCst) {
            sleep(1.0).await;
            messages::Local::new("Hello!").send_local_broadcast(true);
        }
    });
}
