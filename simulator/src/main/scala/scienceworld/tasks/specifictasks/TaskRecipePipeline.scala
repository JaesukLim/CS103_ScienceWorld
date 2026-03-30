package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.containers.MetalPot
import scienceworld.objects.devices.{Sink, Stove}
import scienceworld.objects.substance.food.{Banana, Flour, Jam, Peanut}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMakeIsolatedRoom, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalObjectsInSingleContainer}

import scala.collection.mutable.ArrayBuffer

class TaskRecipePipeline(val mode:String = TaskRecipePipeline.MODE_RECIPE_TINY) extends TaskParametric {
  import TaskRecipePipeline._

  override val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")
  override val isVisibleInTaskList:Boolean = mode != TaskRecipePipeline.MODE_RECIPE_UNSEEN

  private val miniRoomName = "test kitchen"
  private val seenFlourRooms = Array("living room", "bedroom", "art studio")
  private val unseenFlourRooms = Array("living room", "bedroom")
  private val seenFinishingRooms = Array("art studio", "workshop")
  private val unseenFinishingRooms = Array("art studio", "workshop")

  private val tinyRecipes = Array(
    RecipeSpec("bread", Array.empty[String]),
    RecipeSpec("jam sandwich", Array("jam")),
    RecipeSpec("peanut butter sandwich", Array("peanut"))
  )

  private val seenRecipes = Array(
    RecipeSpec("jam sandwich", Array("jam")),
    RecipeSpec("peanut butter sandwich", Array("peanut")),
    RecipeSpec("banana sandwich", Array("banana"))
  )

  private val unseenRecipes = Array(
    RecipeSpec("peanut butter with jam sandwich", Array("peanut", "jam")),
    RecipeSpec("peanut butter with banana sandwich", Array("peanut", "banana"))
  )

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  if (mode == TaskRecipePipeline.MODE_RECIPE_TINY) {
    for (recipe <- tinyRecipes) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = miniRoomName,
        finishingRooms = Array.fill(recipe.finishingIngredients.length)(miniRoomName),
        heatRoom = miniRoomName,
        isMini = true
      ))
    }
  } else if (mode == TaskRecipePipeline.MODE_RECIPE_SEEN) {
    for (recipe <- seenRecipes; flourRoom <- seenFlourRooms; finishingRooms <- finishingRoomAssignments(recipe.finishingIngredients.length, flourRoom, seenFinishingRooms)) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = flourRoom,
        finishingRooms = finishingRooms,
        heatRoom = "kitchen",
        isMini = false
      ))
    }
  } else {
    for (recipe <- unseenRecipes; flourRoom <- unseenFlourRooms; finishingRooms <- finishingRoomAssignments(recipe.finishingIngredients.length, flourRoom, unseenFinishingRooms)) {
      combinations.append(makeRecipeModifiers(
        recipe = recipe,
        flourRoom = flourRoom,
        finishingRooms = finishingRooms,
        heatRoom = "kitchen",
        isMini = false
      ))
    }
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def makeRecipeModifiers(
    recipe:RecipeSpec,
    flourRoom:String,
    finishingRooms:Array[String],
    heatRoom:String,
    isMini:Boolean
  ): Array[TaskModifier] = {
    val modifiers = new ArrayBuffer[TaskModifier]()

    if (isMini) {
      modifiers.append(new TaskMakeIsolatedRoom(miniRoomName))
      modifiers.append(new TaskObject("sink", Some(new Sink(None)), miniRoomName, Array.empty[String], forceAdd = true))
      modifiers.append(new TaskObject("stove", Some(new Stove()), miniRoomName, Array.empty[String], forceAdd = true))
    } else {
      modifiers.append(new TaskObject("sink", Some(new Sink(None)), heatRoom, Array.empty[String]))
      modifiers.append(new TaskObject("stove", Some(new Stove()), heatRoom, Array.empty[String]))
    }

    modifiers.append(new TaskObject("metal pot", Some(new MetalPot()), heatRoom, Array.empty[String], forceAdd = true))
    modifiers.append(new TaskObject("flour", Some(new Flour()), flourRoom, Array.empty[String], forceAdd = true))

    for ((ingredientName, roomName) <- recipe.finishingIngredients.zip(finishingRooms)) {
      modifiers.append(new TaskObject(ingredientName, Some(mkIngredientObject(ingredientName)), roomName, Array.empty[String], forceAdd = true))
    }

    modifiers.append(new TaskValueStr("targetName", recipe.targetName))
    modifiers.append(new TaskValueStr("flourRoom", flourRoom))
    modifiers.append(new TaskValueStr("heatRoom", heatRoom))
    modifiers.append(new TaskValueStr("finishingIngredients", recipe.finishingIngredients.mkString(",")))
    modifiers.append(new TaskValueStr("finishingRooms", finishingRooms.mkString(",")))
    modifiers.append(new TaskValueBool("isMini", isMini))
    modifiers.append(new TaskValueStr("recipe_corpus", SHARED_RECIPE_CORPUS_SERIALIZED))

    modifiers.toArray
  }

  private def finishingRoomAssignments(numIngredients:Int, flourRoom:String, candidateRooms:Array[String]): Array[Array[String]] = {
    if (numIngredients == 0) return Array(Array.empty[String])

    val usableRooms = candidateRooms.filterNot(_ == flourRoom)
    if (usableRooms.isEmpty) {
      return Array(Array.fill(numIngredients)(flourRoom))
    }

    if (numIngredients == 1) {
      return usableRooms.map(room => Array(room))
    }

    val permutations = usableRooms.permutations.map(_.take(numIngredients).toArray).toArray
    if (permutations.isEmpty) Array(Array.fill(numIngredients)(usableRooms.head))
    else permutations
  }

  private def mkIngredientObject(name:String): EnvObject = {
    name match {
      case "jam" => new Jam()
      case "peanut" => new Peanut()
      case "banana" => new Banana()
      case _ => new Jam()
    }
  }

  private def parseCSV(value:String): Array[String] = {
    if (value.trim.isEmpty) Array.empty[String]
    else value.split(",").map(_.trim).filter(_.nonEmpty)
  }

  private def runModifiers(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) return (false, "ERROR: Error running one or more modifiers while setting up task environment.")
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent): (Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (false, "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ").")
    }
    this.runModifiers(this.getCombination(combinationNum), universe, agent)
  }

  private def mkDescription(isMini:Boolean): String = {
    if (isMini) {
      "Your task is to follow the recipe, prepare the final dish, and focus on it. Everything you need is in this room."
    } else if (mode == TaskRecipePipeline.MODE_RECIPE_SEEN) {
      "Your task is to follow the recipe, gather ingredients from around the house, use the kitchen to cook the dish, and then focus on the final result."
    } else {
      "Your task is to follow the recipe, search for ingredients, use the kitchen to cook the dish, and then focus on the final result."
    }
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val targetName = this.getTaskValueStr(modifiers, "targetName").get
    val flourRoom = this.getTaskValueStr(modifiers, "flourRoom").get
    val finishingIngredients = parseCSV(this.getTaskValueStr(modifiers, "finishingIngredients").getOrElse(""))
    val heatRoom = this.getTaskValueStr(modifiers, "heatRoom").get
    val isMini = this.getTaskValueBool(modifiers, "isMini").getOrElse(false)

    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalFind(objectName = targetName, failIfWrong = true, description = "focus on the " + targetName))

    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "metal pot", _isOptional = true, description = "find the metal pot"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "flour", _isOptional = true, description = "find the flour"))
    gSequenceUnordered.append(new GoalObjectsInSingleContainer(objectNames = Array("flour", "water"), _isOptional = true, description = "put flour and water together in one container"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "dough", _isOptional = true, description = "make dough"))
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "bread", _isOptional = true, description = "make bread"))
    if (finishingIngredients.nonEmpty) {
      gSequenceUnordered.append(new GoalObjectsInSingleContainer(objectNames = Array("bread") ++ finishingIngredients, _isOptional = true, description = "combine bread with the finishing ingredients"))
    }
    for (ingredientName <- finishingIngredients) {
      gSequenceUnordered.append(new GoalInRoomWithObject(objectName = ingredientName, _isOptional = true, description = "find the " + ingredientName))
    }
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = "stove", _isOptional = true, description = "use the stove in " + heatRoom))
    if (!isMini && flourRoom != heatRoom) {
      gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))
    }

    val description = mkDescription(isMini)
    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = this.setupGoals(this.getCombination(combinationNum), combinationNum)

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array.empty[String])
  }
}

object TaskRecipePipeline {
  val MODE_RECIPE_TINY = "recipe pipeline tiny"
  val MODE_RECIPE_SEEN = "recipe pipeline seen"
  val MODE_RECIPE_UNSEEN = "recipe pipeline unseen"
  val RECIPE_DOC_SEPARATOR = "\n<<RECIPE_DOC_SEPARATOR>>\n"

  case class RecipeSpec(targetName:String, finishingIngredients:Array[String])
  case class CorpusRecipeSpec(productName:String, ingredientsPhrase:String, actions:Array[String], docCount:Int)

  private val SHARED_RECIPE_SPECS:Array[CorpusRecipeSpec] = Array(
    CorpusRecipeSpec("bread", "a metal pot, flour, and water", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread"), 3),
    CorpusRecipeSpec("dough", "a metal pot, flour, and water", Array("mix the water and flour in the metal pot until dough forms", "keep the dough together in the metal pot until the mixture settles"), 2),
    CorpusRecipeSpec("jam sandwich", "a metal pot, flour, water, and jam", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "mix the bread and jam in the metal pot until a jam sandwich appears"), 3),
    CorpusRecipeSpec("peanut butter sandwich", "a metal pot, flour, water, and peanuts", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "mix the bread and peanuts in the metal pot until a peanut butter sandwich appears"), 3),
    CorpusRecipeSpec("banana sandwich", "a metal pot, flour, water, and a banana", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "mix the bread and banana in the metal pot until a banana sandwich appears"), 3),
    CorpusRecipeSpec("peanut butter with jam sandwich", "a metal pot, flour, water, peanuts, and jam", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "mix the bread, peanuts, and jam in the metal pot until a peanut butter with jam sandwich appears"), 3),
    CorpusRecipeSpec("peanut butter with banana sandwich", "a metal pot, flour, water, peanuts, and a banana", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "mix the bread, peanuts, and banana in the metal pot until a peanut butter with banana sandwich appears"), 3),
    CorpusRecipeSpec("baked potato", "a potato and a stove", Array("place the potato on the stove so it starts heating", "wait until the potato becomes a baked potato"), 1),
    CorpusRecipeSpec("burnt potato", "a potato and a stove", Array("place the potato on the stove so it starts heating", "keep heating the baked potato until it becomes a burnt potato"), 1),
    CorpusRecipeSpec("toasted marshmallow", "a marshmallow and a stove", Array("place the marshmallow on the stove so it starts heating", "wait until the marshmallow becomes a toasted marshmallow"), 1),
    CorpusRecipeSpec("burnt marshmallow", "a marshmallow and a stove", Array("place the marshmallow on the stove so it starts heating", "keep heating the toasted marshmallow until it becomes a burnt marshmallow"), 1),
    CorpusRecipeSpec("burnt bread", "a metal pot, flour, water, and a stove", Array("mix the water and flour in the metal pot until dough forms", "heat the dough on a stove until it becomes bread", "keep heating the bread until it becomes burnt bread"), 1),
    CorpusRecipeSpec("smores", "a metal pot, chocolate, and a marshmallow", Array("put the chocolate and marshmallow into the same metal pot", "mix the metal pot until smores appear"), 2),
    CorpusRecipeSpec("mixed nuts", "a metal pot, peanuts, almonds, and cashews", Array("put the peanuts, almonds, and cashews into the same metal pot", "mix the metal pot until mixed nuts appear"), 2),
    CorpusRecipeSpec("fruit salad", "a metal pot, an apple, an orange, and a banana", Array("put the apple, orange, and banana into the same metal pot", "mix the metal pot until fruit salad appears"), 2),
    CorpusRecipeSpec("salt water", "a glass cup, sodium chloride, and water", Array("put the water and sodium chloride into the same glass cup", "mix the glass cup until salt water appears"), 2),
    CorpusRecipeSpec("soapy water", "a glass cup, soap, and water", Array("put the water and soap into the same glass cup", "mix the glass cup until soapy water appears"), 2),
    CorpusRecipeSpec("sugar water", "a glass cup, sugar, and water", Array("put the water and sugar into the same glass cup", "mix the glass cup until sugar water appears"), 2),
    CorpusRecipeSpec("sodium acetate", "a glass cup, acetic acid, and sodium bicarbonate", Array("put the acetic acid and sodium bicarbonate into the same glass cup", "mix the glass cup until sodium acetate appears"), 2),
    CorpusRecipeSpec("rust", "a metal pot, an iron block, and water", Array("put the iron block and water into the same metal pot", "mix the metal pot until rust appears"), 1),
    CorpusRecipeSpec("red paper", "a container, paper, and red paint", Array("put the paper and red paint into the same container", "mix the container until red paper appears"), 2),
    CorpusRecipeSpec("green paper", "a container, paper, and green paint", Array("put the paper and green paint into the same container", "mix the container until green paper appears"), 2),
    CorpusRecipeSpec("blue paper", "a container, paper, and blue paint", Array("put the paper and blue paint into the same container", "mix the container until blue paper appears"), 2),
    CorpusRecipeSpec("orange paper", "a container, paper, and orange paint", Array("put the paper and orange paint into the same container", "mix the container until orange paper appears"), 2),
    CorpusRecipeSpec("violet paper", "a container, paper, and violet paint", Array("put the paper and violet paint into the same container", "mix the container until violet paper appears"), 2)
  )

  private def renderCanonicalDoc(spec:CorpusRecipeSpec): String = {
    val lines = Array(
      "Recipe card for " + spec.productName + ".",
      "Gather " + spec.ingredientsPhrase + "."
    ) ++ spec.actions.map(action => action.capitalize + ".") ++ Array(
      "Focus on the " + spec.productName + " when it is ready."
    )

    lines.mkString("\n")
  }

  private def renderHardPositiveDoc(spec:CorpusRecipeSpec): String = {
    val actionLines = spec.actions.zipWithIndex.map {
      case (action, 0) => "Start by " + action + "."
      case (action, _) => "Then " + action + "."
    }

    val lines = Array(
      "Kitchen note for " + spec.productName + ".",
      "Start with " + spec.ingredientsPhrase + "."
    ) ++ actionLines ++ Array(
      "Focus on the " + spec.productName + " once the transformation is complete."
    )

    lines.mkString("\n")
  }

  private def renderHardPositiveDoc2(spec:CorpusRecipeSpec): String = {
    val actionLines = spec.actions.zipWithIndex.map {
      case (action, 0) => "Use the same ingredients to " + action + "."
      case (action, _) => "After that, " + action + "."
    }

    val lines = Array(
      "Lab memo for " + spec.productName + ".",
      "Collect " + spec.ingredientsPhrase + " before starting."
    ) ++ actionLines ++ Array(
      "End by focusing on the " + spec.productName + "."
    )

    lines.mkString("\n")
  }

  val SHARED_RECIPE_CORPUS:Array[String] = SHARED_RECIPE_SPECS.flatMap { spec =>
    spec.docCount match {
      case 1 => Array(renderCanonicalDoc(spec))
      case 2 => Array(renderCanonicalDoc(spec), renderHardPositiveDoc(spec))
      case _ => Array(renderCanonicalDoc(spec), renderHardPositiveDoc(spec), renderHardPositiveDoc2(spec))
    }
  }

  val SHARED_RECIPE_CORPUS_SERIALIZED:String = SHARED_RECIPE_CORPUS.mkString(RECIPE_DOC_SEPARATOR)

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_TINY))
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_SEEN))
    taskMaker.addTask(new TaskRecipePipeline(mode = MODE_RECIPE_UNSEEN))
  }
}
