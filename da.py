def training_and_evaluating(model, optimizer, loss_type, device):
    for epoch in range(1, epochs + 1):
        def print_epoch_results(engine, type=''):
            result = "\t".join([f"{i}:{j}" for i, j in engine.state.metrics.items()])
            print(f"Results after {type} epoch:", result)

        def one_epoch(engine):
            # evaluate on trainig and validation datasets
            train_eval.run(train_loader)
            valid_eval.run(validation_loader)

            # extract metrics from engine
            validation_results = valid_eval.state.metrics
            train_results = train_eval.state.metrics

            # metrics are stored into dictionary
            train_accuracy.append(train_results['Accuracy'])
            validation_accuracy.append(validation_results['Accuracy'])
            train_loss.append(train_results['Loss'])
            validation_loss.append(validation_results['Loss'])

        # dynamicalli scetch plots
        draw_plots(train_accuracy, validation_accuracy,
                   train_loss, validation_loss)

        # clear the grid before new epoch 
        def clear_output_handler(engine):
            clear_output(wait=True)

        trainer = create_supervised_trainer(model, optimizer, loss_type, device)

        # choose metrics to show
        metrics_for_task = {
            "Accuracy": Accuracy(),
            "Precision": Precision().mean(),
            "Recall": Recall().mean(),
            "Loss": Loss(loss_type)
        }

        train_eval = create_supervised_evaluator(model, device=device, metrics=metrics_for_task)
        valid_eval = create_supervised_evaluator(model, device=device, metrics=metrics_for_task)

        # start new epoch
        trainer.add_event_handler(Events.EPOCH_COMPLETED, one_epoch)

        # print train and validation results
        train_eval.add_event_handler(Events.EPOCH_COMPLETED, print_epoch_results, type_train_eval="Train")
        valid_eval.add_event_handler(Events.EPOCH_COMPLETED, print_epoch_results, type_train_eval="Validation")

        print(f'Epoch {epoch} from {epochs}')
        # train model
        trainer.run(train_loader)