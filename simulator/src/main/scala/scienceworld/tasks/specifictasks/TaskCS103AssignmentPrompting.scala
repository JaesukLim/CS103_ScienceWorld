package scienceworld.tasks.specifictasks

import scienceworld.goldagent.PathFinder
import scienceworld.objects.agent.Agent
import scienceworld.objects.devices.{Shovel, Thermometer}
import scienceworld.objects.document.Paper
import scienceworld.objects.electricalcomponent.Battery
import scienceworld.objects.substance.paint.BluePaint
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalMoveToLocation, GoalMoveToNewLocation, GoalSpecificObjectInDirectContainer}
import scienceworld.tasks.{Task, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}

import scala.collection.mutable.ArrayBuffer

class TaskCS103AssignmentPrompting(
  val mode:String = TaskCS103AssignmentPrompting.MODE_ASSIGNMENT_1_PROMPTING
) extends TaskParametric {
  val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  addVariation(new Thermometer(), "kitchen", requiresPickup = false, destinationLocation = "")
  addVariation(new Battery(), "workshop", requiresPickup = true, destinationLocation = "")
  addVariation(new BluePaint(), "art studio", requiresPickup = true, destinationLocation = "living room")
  addVariation(new Paper(), "bedroom", requiresPickup = true, destinationLocation = "hallway")
  addVariation(new Shovel(), "outside", requiresPickup = true, destinationLocation = "greenhouse")

  private def addVariation(
    exampleObject:EnvObject,
    sourceLocation:String,
    requiresPickup:Boolean,
    destinationLocation:String
  ): Unit = {
    combinations.append(Array(
      new TaskObject(
        name = exampleObject.name,
        exampleInstance = Some(exampleObject),
        roomToGenerateIn = sourceLocation,
        possibleContainerNames = Array.empty[String]
      ),
      new TaskValueStr("objectName", exampleObject.name),
      new TaskValueStr("sourceLocation", sourceLocation),
      new TaskValueStr("destinationLocation", destinationLocation),
      new TaskValueBool("requiresPickup", requiresPickup)
    ))
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def setupCombination(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) {
        return (false, "ERROR: Failed to apply one or more assignment prompting modifiers.")
      }
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent):(Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (
        false,
        "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ")."
      )
    }
    this.setupCombination(this.getCombination(combinationNum), universe, agent)
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val objectName = this.getTaskValueStr(modifiers, "objectName").get
    val sourceLocation = this.getTaskValueStr(modifiers, "sourceLocation").get
    val destinationLocation = this.getTaskValueStr(modifiers, "destinationLocation").get
    val requiresPickup = this.getTaskValueBool(modifiers, "requiresPickup").get

    val gSequence = new ArrayBuffer[Goal]
    val gSequenceUnordered = new ArrayBuffer[Goal]

    gSequence.append(new GoalMoveToLocation(sourceLocation, description = "move to the source location"))

    if (requiresPickup) {
      gSequence.append(
        new GoalSpecificObjectInDirectContainer(
          containerName = "inventory",
          validObjectNames = Array(objectName),
          description = "pick up the target object"
        )
      )
    }

    if (destinationLocation.nonEmpty) {
      gSequence.append(new GoalMoveToLocation(destinationLocation, description = "move to the destination location"))
    }

    gSequence.append(new GoalFind(objectName = objectName, failIfWrong = true, description = "focus on the target object"))

    gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move away from the starting location"))

    var description = "This is CS103 Assignment 1 (Prompting). "
    description += "Go to the " + sourceLocation + ". "
    if (requiresPickup) {
      description += "Pick up the " + objectName + ". "
    }
    if (destinationLocation.nonEmpty) {
      description += "Then go to the " + destinationLocation + ". "
    }
    description += "Finally, focus on the " + objectName + "."

    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = {
    this.setupGoals(this.getCombination(combinationNum), combinationNum)
  }

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface):(Boolean, Array[String]) = {
    val universe = runner.agentInterface.get.universe
    val agent = runner.agentInterface.get.agent

    val objectName = this.getTaskValueStr(modifiers, "objectName").get
    val sourceLocation = this.getTaskValueStr(modifiers, "sourceLocation").get
    val destinationLocation = this.getTaskValueStr(modifiers, "destinationLocation").get
    val requiresPickup = this.getTaskValueBool(modifiers, "requiresPickup").get

    val (_, actionStrsToSource) = PathFinder.createActionSequence(
      universe,
      agent,
      startLocation = getCurrentAgentLocation(runner).name,
      endLocation = sourceLocation
    )
    runActionSequence(actionStrsToSource, runner)
    runAction("look around", runner)

    val visibleTargets = PathFinder.getAllAccessibleEnvObject(queryName = objectName, getCurrentAgentLocation(runner))
    if (visibleTargets.isEmpty) {
      return (false, getActionHistory(runner))
    }
    val target = visibleTargets(0)

    if (requiresPickup) {
      val targetReferent = PathFinder.getObjUniqueReferent(target, getCurrentAgentLocation(runner))
      if (targetReferent.isEmpty) {
        return (false, getActionHistory(runner))
      }
      runAction("pick up " + targetReferent.get, runner)
    }

    if (destinationLocation.nonEmpty) {
      val (_, actionStrsToDestination) = PathFinder.createActionSequence(
        universe,
        agent,
        startLocation = getCurrentAgentLocation(runner).name,
        endLocation = destinationLocation
      )
      runActionSequence(actionStrsToDestination, runner)
      runAction("look around", runner)
    }

    if (requiresPickup) {
      runAction("focus on " + objectName + " in inventory", runner)
    } else {
      val targetReferent = PathFinder.getObjUniqueReferent(target, getCurrentAgentLocation(runner))
      if (targetReferent.isEmpty) {
        return (false, getActionHistory(runner))
      }
      runAction("focus on " + targetReferent.get, runner)
    }

    (true, getActionHistory(runner))
  }
}

object TaskCS103AssignmentPrompting {
  val MODE_ASSIGNMENT_1_PROMPTING = "assignment 1 prompting"

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskCS103AssignmentPrompting())
  }
}
